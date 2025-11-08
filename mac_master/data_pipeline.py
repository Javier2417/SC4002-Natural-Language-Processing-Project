
# === SC4002 Data Pipeline Module ===
# This file contains all Part 0/1 logic.
# Teams can import this file to get all the data loaders and helper functions.

import torch
import torch.nn as nn
import torchtext
from torchtext import data, datasets
import random
import numpy as np

# 1. Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

assert torchtext.__version__.startswith("0.4."), \
    f"Expected torchtext 0.4.x, got {torchtext.__version__}"

# 2. Define Fields (as per PDF)
# We define these at the top level so they can be imported
TEXT = data.Field(tokenize='spacy', 
                  tokenizer_language='en_core_web_sm', 
                  lower=True, 
                  include_lengths=True)
LABEL = data.LabelField()

# 3. Load Dataset and create splits
train_data, test_data = datasets.TREC.splits(TEXT, LABEL, fine_grained=False)

train_dataset, valid_dataset = train_data.split(
    split_ratio=0.8, 
    stratified=True, 
    strata_field='label', 
    random_state=random.seed(SEED)
)

# 4. Build Vocabulary and load GloVe vectors (Q1c)
TEXT.build_vocab(train_dataset,
                 min_freq=1,
                 vectors="glove.6B.300d",
                 unk_init=torch.Tensor.normal_) # Q1c strategy
LABEL.build_vocab(train_dataset)

# 5. Define device and create iterators (Artifacts)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_dataset, valid_dataset, test_data), 
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text), # Sorts by length (good for RNN)
    sort_within_batch=True,
    device=device
)

# 6. Helper Function for Part 2/3 (Artifact)
def create_embedding_layer(freeze=False):
    """
    Loads the pre-trained GloVe vectors from the TEXT vocab
    into a learnable nn.Embedding layer.
    
    This version correctly handles the <pad> token by:
    1. Setting padding_idx in the Embedding constructor.
    2. Manually copying the weights.
    3. Manually zeroing out the padding vector.
    """
    
    # 1. Get info from the vocab
    pretrained_embeddings = TEXT.vocab.vectors
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    
    # 2. Create the embedding layer, specifying the pad_idx
    embedding_layer = nn.Embedding(
        pretrained_embeddings.size(0), 
        pretrained_embeddings.size(1), 
        padding_idx=pad_idx
    )
    
    # 3. Copy the GloVe weights
    embedding_layer.weight.data.copy_(pretrained_embeddings)
    
    # 4. Manually zero out the padding vector
    embedding_layer.weight.data[pad_idx].zero_()
    
    # 5. Set the layer to be learnable
    embedding_layer.weight.requires_grad = not freeze
    
    return embedding_layer


# from data_pipeline import (
#     train_iterator, 
#     valid_iterator, 
#     test_iterator, 
#     train_dataset,
#     valid_dataset,
#     test_data,
#     TEXT, 
#     LABEL, 
#     create_embedding_layer,
#     device,
#     BATCH_SIZE
# )

