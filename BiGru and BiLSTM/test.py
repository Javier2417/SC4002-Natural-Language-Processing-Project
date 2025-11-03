import importlib
import torch
import numpy as np
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(".."))

# --- PASTE THE FULL set_seed FUNCTION HERE ---
# (This is just to be extra sure, as the one in cell 19 should also work)
def set_seed(seed):
    """Sets the random seed for full reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Force deterministic algorithms
    torch.use_deterministic_algorithms(True) 
    
    # Configure CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    # This is often needed for deterministic bmm/RNNs on GPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
# --- END set_seed ---


print("--- TEST 1: RUNNING PIPELINE (FIRST TIME) ---")
set_seed(42)
# We need to re-import data_pipeline to get its initial state
import data_pipeline
first_batch_1 = next(iter(data_pipeline.train_iterator))
unk_vector_1 = data_pipeline.TEXT.vocab.vectors[data_pipeline.TEXT.vocab.stoi[data_pipeline.TEXT.unk_token]]
text_1, _ = first_batch_1.text
labels_1 = first_batch_1.label


print("\n--- TEST 2: RESETTING AND RUNNING PIPELINE (SECOND TIME) ---")
# Reload the module to reset its state, as if we restarted the kernel
importlib.reload(data_pipeline)

# Set the exact same seed again
set_seed(42)

# We must re-import the reloaded iterators
# This re-runs the split, vocab build, and iterator creation
from data_pipeline import train_iterator as train_iterator_2
from data_pipeline import TEXT as TEXT_2

first_batch_2 = next(iter(train_iterator_2))
unk_vector_2 = TEXT_2.vocab.vectors[TEXT_2.vocab.stoi[TEXT_2.unk_token]]
text_2, _ = first_batch_2.text
labels_2 = first_batch_2.label


# --- 3. COMPARISON ---
print("\n--- 3. COMPARING RESULTS ---")
is_text_equal = torch.equal(text_1, text_2)
is_label_equal = torch.equal(labels_1, labels_2)
is_unk_equal = torch.equal(unk_vector_1, unk_vector_2)

print(f"\nFirst batch text tensors are identical:   {is_text_equal}")
print(f"First batch label tensors are identical:  {is_label_equal}")
print(f"<unk> token vectors are identical:        {is_unk_equal}")

if is_text_equal and is_label_equal and is_unk_equal:
    print("\n✅ SUCCESS: The data pipeline is deterministic.")
else:
    print("\n❌ FAILURE: The data pipeline is not deterministic.")