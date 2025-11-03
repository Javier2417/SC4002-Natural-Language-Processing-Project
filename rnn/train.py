import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import config

from data_pipeline import (
    train_iterator, 
    valid_iterator, 
    test_iterator, 
    TEXT, 
    LABEL, 
    create_embedding_layer
)

torch.manual_seed(42)

# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):

        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model when validation loss improves."""
        self.best_weights = model.state_dict().copy()

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, embedding_layer=None):
        super().__init__()

        if embedding_layer is not None:
            self.embedding = embedding_layer
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=TEXT.vocab.stoi[TEXT.pad_token])
        dropout = dropout if n_layers > 1 else 0.0
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        # Take last layer
        output = self.fc(hidden[-1])
        return output

# Create Model and Embeddings
hidden_dim = 128
n_layers = 2
dropout = 0.5
output_dim = len(LABEL.vocab)

print("Creating embedding layer from TEXT.vocab...")
embedding_layer = create_embedding_layer()
embedding_dim = embedding_layer.embedding_dim if hasattr(embedding_layer, 'embedding_dim') else 300

model = RNN(
    vocab_size=len(TEXT.vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    dropout=dropout,
    embedding_layer=embedding_layer
)

# Define Optimizer and Loss Function
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(config.DEVICE)
criterion = criterion.to(config.DEVICE)

# Accuracy Function
def accuracy(preds, y):
    top_pred = preds.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    return correct.float() / y.shape[0] * 100

# Train and Evaluate Functions
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        text, text_lengths = batch.text
        labels = batch.label

        text, text_lengths, labels = text.to(config.DEVICE), text_lengths.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        predictions = model(text, text_lengths)
        loss = criterion(predictions, labels)
        acc = accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            labels = batch.label

            text, text_lengths, labels = text.to(config.DEVICE), text_lengths.to(config.DEVICE), labels.to(config.DEVICE)

            predictions = model(text, text_lengths)
            loss = criterion(predictions, labels)
            acc = accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# Training Loop with Early Stopping
N_EPOCHS = config.N_EPOCHS

# Initialize early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.01, restore_best_weights=True)

# Lists to track training progress
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
epochs_completed = []

print("Starting training with early stopping...")
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    # Store metrics for plotting
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(valid_loss)
    val_accuracies.append(valid_acc)
    epochs_completed.append(epoch + 1)

    print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | Val Loss: {valid_loss:.3f} | Val Acc: {valid_acc:.2f}%')
    
    # Check early stopping
    if early_stopping(valid_loss, model):
        print(f'Early stopping triggered at epoch {epoch+1}')
        print(f'Best validation loss: {early_stopping.best_loss:.3f}')
        break

# Evaluation
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'\nTest Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}%')

# Plot training progress
def plot_training_progress(epochs, train_losses, val_losses, train_accs, val_accs):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training progress plot saved as 'training_progress.png'")

# Create and save the training progress plot
plot_training_progress(epochs_completed, train_losses, val_losses, train_accuracies, val_accuracies)

# Save Model and Vocab
torch.save(model.state_dict(), 'rnn_model.pt')
print("Model saved to rnn_model.pt")

torch.save({
    'TEXT_vocab': TEXT.vocab,
    'LABEL_vocab': LABEL.vocab
}, 'vocab.pt')

print("Vocabulary saved to vocab.pt")