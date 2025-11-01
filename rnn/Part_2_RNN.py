
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import urllib.request
import os
import config

# Set the random seed for reproducibility
torch.manual_seed(42)

# Download and parse TREC dataset
def download_trec():
    base_url = "https://cogcomp.seas.upenn.edu/Data/QA/QC/"
    files = ['train_5500.label', 'TREC_10.label']
    
    for file in files:
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, file)
    
    return files

def parse_trec_file(filepath):
    texts = []
    labels = []
    with open(filepath, 'r', encoding='latin-1') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    label, text = parts
                    coarse_label = label.split(':')[0]
                    texts.append(text.lower().split())
                    labels.append(coarse_label)
    return texts, labels

# Load data
download_trec()
train_texts, train_labels = parse_trec_file('train_5500.label')
test_texts, test_labels = parse_trec_file('TREC_10.label')

# Split training data into train and validation
train_texts, valid_texts, train_labels, valid_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

# Build vocabulary
def build_vocab(texts, min_freq=2):
    word_freq = Counter()
    for text in texts:
        word_freq.update(text)
    
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab

def build_label_vocab(labels):
    unique_labels = sorted(set(labels))
    return {label: idx for idx, label in enumerate(unique_labels)}

vocab = build_vocab(train_texts)
label_vocab = build_label_vocab(train_labels)

print(f"Vocabulary size: {len(vocab)}")
print(f"Number of classes: {len(label_vocab)}")
print(f"Classes: {list(label_vocab.keys())}")

# Load GloVe embeddings
def load_glove_embeddings(vocab, embedding_dim=100):
    glove_file = f'glove.6B.{embedding_dim}d.txt'
    
    if not os.path.exists(glove_file):
        print(f"Downloading GloVe embeddings...")
        import gensim.downloader as api
        glove_model = api.load(f'glove-wiki-gigaword-{embedding_dim}')
        
        embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
        embeddings[0] = 0  # padding vector
        
        for word, idx in vocab.items():
            if word in glove_model:
                embeddings[idx] = glove_model[word]
    else:
        embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
        embeddings[0] = 0
        
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab:
                    vector = np.array(values[1:], dtype='float32')
                    embeddings[vocab[word]] = vector
    
    return torch.FloatTensor(embeddings)

# Dataset class
class TRECDataset(Dataset):
    def __init__(self, texts, labels, vocab, label_vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_vocab = label_vocab
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = [self.vocab.get(word, self.vocab['<unk>']) for word in self.texts[idx]]
        label = self.label_vocab[self.labels[idx]]
        return torch.LongTensor(text), label

def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = torch.LongTensor([len(text) for text in texts])
    texts = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    return texts, lengths, labels

# Create datasets and dataloaders
train_dataset = TRECDataset(train_texts, train_labels, vocab, label_vocab)
valid_dataset = TRECDataset(valid_texts, valid_labels, vocab, label_vocab)
test_dataset = TRECDataset(test_texts, test_labels, vocab, label_vocab)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Define the RNN model
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
        
        # Sort by length for packing
        lengths_sorted, sorted_idx = text_lengths.sort(descending=True)
        embedded_sorted = embedded[sorted_idx]
        
        packed_embedded = pack_padded_sequence(embedded_sorted, lengths_sorted.cpu(), batch_first=True)
        packed_output, hidden = self.rnn(packed_embedded)
        
        # Unsort
        _, unsorted_idx = sorted_idx.sort()
        hidden = hidden[:, unsorted_idx, :]
        
        return self.fc(hidden[-1])

# Initialize the model
embedding_dim = 100
hidden_dim = 128
n_layers = 1
dropout = 0.5

print("Loading GloVe embeddings...")
pretrained_embeddings = load_glove_embeddings(vocab, embedding_dim)

model = RNN(len(vocab), embedding_dim, hidden_dim, len(label_vocab), n_layers, dropout, pretrained_embeddings)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Move the model to the specified device
model = model.to(config.DEVICE)
criterion = criterion.to(config.DEVICE)

# Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for texts, lengths, labels in loader:
        texts = texts.to(config.DEVICE)
        lengths = lengths.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()
        predictions = model(texts, lengths)
        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# Evaluation loop
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for texts, lengths, labels in loader:
            texts = texts.to(config.DEVICE)
            lengths = lengths.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            predictions = model(texts, lengths)
            loss = criterion(predictions, labels)
            acc = accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# Function to calculate accuracy
def accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0] * 100

# Main training process
N_EPOCHS = config.N_EPOCHS

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    
    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc:.2f}%')

# Test the model
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

# Save the model
torch.save(model.state_dict(), 'rnn_model.pt')
print("Model saved to rnn_model.pt")


# Save vocabulary
import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'label_vocab': label_vocab}, f)
print("Vocabulary saved to vocab.pkl")