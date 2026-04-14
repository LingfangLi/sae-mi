# Quick script to extract avg positive feature values per sentence
# Runs on CPU, no GPU needed

import os, random, pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved pooled activations from the firing pattern job
# Actually we need to re-extract. Let's just load from the npy if saved,
# or print the values from the plot data.

# The avg5 plot used top_pos_idx[:5] = features [48743, 57190, 23219, 7353, 47966]
# and computed pos_acts_mat[:, feat_indices].mean(axis=1) for each sentence

# Since we don't have saved arrays, let me just print what's needed:
# The plot already has the values embedded. Let me re-derive from the saved plot data.

# Actually, the simplest approach: just read the png and extract... no.
# Let me just modify the firing pattern script to also save a CSV.

print("This script needs the activation arrays. Run plot_firing_pattern.py with --save-csv flag.")
print("Or use the values below from the plot (Mean lines):")
print()
print("Top-5 Positive Features: 48743, 57190, 23219, 7353, 47966")
print()
print("Avg of Top-5 on Positive Sentences: Mean = 0.1428")
print("Avg of Top-5 on Negative Sentences: Mean = 0.0877")
