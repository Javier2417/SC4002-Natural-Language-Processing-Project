import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
BATCH_SIZE = 64
N_EPOCHS = 10
LEARNING_RATE = 0.001

# Model hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
N_LAYERS = 1
DROPOUT = 0.3

# Vocabulary settings
MIN_FREQ = 2

# Paths
DATA_DIR = '../data'
MODEL_SAVE_PATH = '../models/rnn_model.pt'

print(f"Using device: {DEVICE}")