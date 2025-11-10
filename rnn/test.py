import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from collections import defaultdict
from data_pipeline import create_embedding_layer


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pretrained_embeddings=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        
        lengths_sorted, sorted_idx = text_lengths.sort(descending=True)
        embedded_sorted = embedded[sorted_idx]
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded_sorted, lengths_sorted.cpu(), batch_first=True)
        packed_output, hidden = self.rnn(packed_embedded)
        
        _, unsorted_idx = sorted_idx.sort()
        hidden = hidden[:, unsorted_idx, :]
        
        return self.fc(hidden[-1])

# Prediction Question
def predict_question(model, question, vocab, label_vocab, device):
    model.eval()
    
    tokens = question.lower().split()
    indices = [vocab.stoi.get(word, vocab.stoi['<unk>']) for word in tokens]
    
    text_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    length_tensor = torch.LongTensor([len(indices)]).to(device)
    
    with torch.no_grad():
        output = model(text_tensor, length_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
    
    idx_to_label = {i: label for i, label in enumerate(label_vocab.itos)}
    predicted_label = idx_to_label[predicted_class]
    
    prob_dict = {idx_to_label[i]: probabilities[0][i].item() for i in range(len(label_vocab))}
    
    return predicted_label, prob_dict

def save_vocab(vocab, label_vocab, filepath='vocab.pt'):
    # Save as torch dictionary
    torch.save({'vocab': vocab, 'label_vocab': label_vocab}, filepath)

def load_vocab(filepath='vocab.pt'):
    # Load from torch file
    data = torch.load('vocab.pt', map_location='cpu')
    TEXT_vocab = data['TEXT_vocab']
    LABEL_vocab = data['LABEL_vocab']
    return TEXT_vocab, LABEL_vocab

# Per-topic Accuracy Evaluation
def evaluate_per_class(model, iterator, label_vocab, device):
    model.eval()
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            labels = batch.label
            text, text_lengths, labels = text.to(device), text_lengths.to(device), labels.to(device)

            predictions = model(text, text_lengths)
            top_pred = predictions.argmax(1)

            for label, pred in zip(labels, top_pred):
                label_name = label_vocab.itos[label]
                total_counts[label_name] += 1
                if pred == label:
                    correct_counts[label_name] += 1

    per_class_acc = {label: 100 * correct_counts[label] / total_counts[label] 
                     for label in total_counts.keys()}

    return per_class_acc

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load vocab
    try:
        vocab, label_vocab = load_vocab('vocab.pt')
        print("Vocabulary loaded successfully!")
    except FileNotFoundError:
        print("Error: vocab.pt not found. Please train the model first and save the vocabulary.")
        exit(1)

    embedding_layer = create_embedding_layer()
    embedding_dim = embedding_layer.embedding_dim if hasattr(embedding_layer, 'embedding_dim') else 300
    hidden_dim = 128
    n_layers = 2
    dropout = 0.5

    # Load model
    model = RNN(len(vocab), embedding_dim, hidden_dim, len(label_vocab), n_layers, dropout)
    try:
        model.load_state_dict(torch.load('rnn_model.pt', map_location=device), strict = False)
        model.to(device)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: rnn_model.pt not found. Please train the model first.")
        exit(1)

    # Topic-wise Accuracy
    from data_pipeline import test_iterator, LABEL

    per_topic_accuracy = evaluate_per_class(model, test_iterator, LABEL.vocab, device)

    print("\n----- Topic-wise Accuracy on Test Set -----")
    for topic, acc in per_topic_accuracy.items():
        print(f"{topic}: {acc:.2f}%")

    # Sample Question Testing
    test_questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "How many people live in China?",
        "When did World War II end?",
        "Where is the Eiffel Tower located?",
        "What does CPU stand for?"
    ]
    
    print("\n" + "="*70)
    print("Testing RNN Model with Sample Questions")
    print("="*70 + "\n")
    
    for question in test_questions:
        predicted_label, probabilities = predict_question(model, question, vocab, label_vocab, device)
        print(f"Question: {question}")
        print(f"Predicted Category: {predicted_label}")
        print("Probabilities:")
        for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {prob*100:.2f}%")
        print("-" * 70 + "\n")

    # Interactive Mode
    print("\n" + "="*70)
    print("Interactive Mode - Enter your own questions (type 'quit' to exit)")
    print("="*70 + "\n")
    
    while True:
        user_input = input("Enter a question: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        if not user_input:
            continue
        predicted_label, probabilities = predict_question(model, user_input, vocab, label_vocab, device)
        print(f"\nPredicted Category: {predicted_label}")
        print("Probabilities:")
        for label, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {prob*100:.2f}%")
        print("\n" + "-" * 70 + "\n")