"""
Gemma 2 2B SST-2: MI+LR feature selection on saved activation matrix.
"""
import os
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_mlp")
LAYER = 13
TOP_K = 150

print(f"Loading layer {LAYER} activation matrix...")
activations = np.load(os.path.join(RESULTS_DIR, f"layer_{LAYER}_activations.npy"))
labels = np.load(os.path.join(RESULTS_DIR, f"layer_{LAYER}_labels.npy"))
print(f"  Shape: {activations.shape}, labels: {labels.shape}")

n_samples, n_features = activations.shape

# ======================================================================
# MI scores
# ======================================================================
print(f"\nComputing MI scores for {n_features} features...")
mi_scores = mutual_info_classif(activations, labels, random_state=42, n_neighbors=5)
top_mi_indices = np.argsort(mi_scores)[::-1][:TOP_K]
print(f"  MI top-{TOP_K}: max MI={mi_scores[top_mi_indices[0]]:.4f}, "
      f"min MI={mi_scores[top_mi_indices[-1]]:.4f}")

# ======================================================================
# LR weights
# ======================================================================
print(f"\nFitting LogisticRegression...")
lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42, solver='liblinear')
lr.fit(activations, labels)
lr_weights = np.abs(lr.coef_[0])
top_lr_indices = np.argsort(lr_weights)[::-1][:TOP_K]
print(f"  LR top-{TOP_K}: max |w|={lr_weights[top_lr_indices[0]]:.4f}, "
      f"min |w|={lr_weights[top_lr_indices[-1]]:.4f}")
print(f"  LR accuracy: {lr.score(activations, labels):.4f}")

# ======================================================================
# Union + sparsity filter
# ======================================================================
union_set = set(top_mi_indices.tolist()) | set(top_lr_indices.tolist())
overlap = set(top_mi_indices.tolist()) & set(top_lr_indices.tolist())
union_indices = np.array(sorted(union_set))
print(f"\nUnion: {len(union_indices)} (overlap: {len(overlap)})")

# Sparsity filter: 0.01 <= act_rate < 0.80
act_rates = (activations[:, union_indices] > 0).sum(axis=0) / n_samples

mask = (act_rates >= 0.01) & (act_rates < 0.80)
filtered_features = union_indices[mask]
filtered_mi = mi_scores[filtered_features]
filtered_lr = lr_weights[filtered_features]
filtered_rates = act_rates[mask]

print(f"After sparsity filter: {len(filtered_features)} features")
print(f"  Activation rate distribution:")
bins = [(0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.40), (0.40, 0.60), (0.60, 0.80)]
for lo, hi in bins:
    n = ((filtered_rates >= lo) & (filtered_rates < hi)).sum()
    print(f"    [{lo:.2f}, {hi:.2f}): {n}")

# ======================================================================
# Save
# ======================================================================
out_path = os.path.join(RESULTS_DIR, f"layer_{LAYER}_feature_selection.npz")
np.savez(
    out_path,
    mi_scores=mi_scores,
    lr_weights=lr_weights,
    top_mi_indices=top_mi_indices,
    top_lr_indices=top_lr_indices,
    union_indices=union_indices,
    filtered_features=filtered_features,
    filtered_mi=filtered_mi,
    filtered_lr_weights=filtered_lr,
    filtered_act_rates=filtered_rates,
)
print(f"\nSaved to: {out_path}")
