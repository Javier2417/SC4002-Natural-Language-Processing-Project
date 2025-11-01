import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pickle


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
        
        from torch.nn.utils.rnn import pack_padded_sequence
        packed_embedded = pack_padded_sequence(embedded_sorted, lengths_sorted.cpu(), batch_first=True)
        packed_output, hidden = self.rnn(packed_embedded)
        
        
        _, unsorted_idx = sorted_idx.sort()
        hidden = hidden[:, unsorted_idx, :]
        
        return self.fc(hidden[-1])

def predict_question(model, question, vocab, label_vocab, device):

    model.eval()
    
    
    tokens = question.lower().split()
    indices = [vocab.get(word, vocab['<unk>']) for word in tokens]
    
    
    text_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)  # Add batch dimension
    length_tensor = torch.LongTensor([len(indices)]).to(device)
    
    
    with torch.no_grad():
        output = model(text_tensor, length_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
    
    
    idx_to_label = {v: k for k, v in label_vocab.items()}
    predicted_label = idx_to_label[predicted_class]
    
    
    prob_dict = {idx_to_label[i]: probabilities[0][i].item() for i in range(len(label_vocab))}
    
    return predicted_label, prob_dict


def save_vocab(vocab, label_vocab, filepath='vocab.pkl'):

    with open(filepath, 'wb') as f:
        pickle.dump({'vocab': vocab, 'label_vocab': label_vocab}, f)

def load_vocab(filepath='vocab.pkl'):

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['vocab'], data['label_vocab']

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    try:
        vocab, label_vocab = load_vocab('vocab.pkl')
        print("Vocabulary loaded successfully!")
    except FileNotFoundError:
        print("Error: vocab.pkl not found. Please train the model first and save the vocabulary.")
        exit(1)
    

    embedding_dim = 100
    hidden_dim = 256
    n_layers = 2
    dropout = 0.5
    

    model = RNN(len(vocab), embedding_dim, hidden_dim, len(label_vocab), n_layers, dropout)
    

    try:
        model.load_state_dict(torch.load('rnn_model.pt', map_location=device))
        model.to(device)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Error: rnn_model.pt not found. Please train the model first.")
        exit(1)
    

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