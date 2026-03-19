"""
Dual-Alignment Analysis: Cross-model SAE feature comparison.
Computes behavioral similarity (Jaccard + conditional Pearson) and
semantic similarity (SentenceTransformer on Groq descriptions),
then generates the Money Plot.

Usage:
  python dual_alignment_analysis.py --model_a gpt2 --model_b qwen25
  python dual_alignment_analysis.py --model_a gpt2 --model_b gemma2
"""

import os
import json
import argparse
import numpy as np
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ======================================================================
# Paths
# ======================================================================
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    "gpt2": {
        "name": "GPT-2 Small",
        "layer": 6,
        "results_dir": os.path.join(BASE, "GPT2", "results_mlp"),
        "n_features": 32768,
    },
    "qwen25": {
        "name": "Qwen2.5 1.5B",
        "layer": 15,
        "results_dir": os.path.join(BASE, "Qwen2.5 1.5B", "results_mlp"),
        "n_features": 65536,
    },
    "gemma2": {
        "name": "Gemma 2 2B",
        "layer": None,  # will be set after probing completes
        "results_dir": os.path.join(BASE, "Gemma 2b", "results_mlp"),
        "n_features": 16384,
    },
}

OUTPUT_DIR = os.path.join(BASE, "dual_alignment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_data(model_key):
    """Load activation matrix, feature selection, and descriptions for a model."""
    cfg = MODEL_CONFIGS[model_key]
    layer = cfg["layer"]
    rdir = cfg["results_dir"]

    print(f"Loading {cfg['name']} layer {layer} data...")

    activations = np.load(os.path.join(rdir, f"layer_{layer}_activations.npy"))
    print(f"  Activations: {activations.shape}")

    fs = np.load(os.path.join(rdir, f"layer_{layer}_feature_selection.npz"))
    filtered_features = fs["filtered_features"]
    print(f"  Filtered features: {len(filtered_features)}")

    desc_path = os.path.join(rdir, f"layer_{layer}_descriptions.json")
    with open(desc_path, "r") as f:
        descriptions = json.load(f)
    print(f"  Descriptions: {len(descriptions)}")

    return activations, filtered_features, descriptions


# ======================================================================
# Behavioral Similarity
# ======================================================================
def compute_jaccard(act_a, act_b):
    """Jaccard index on binary activation patterns."""
    bin_a = (act_a > 0).astype(float)
    bin_b = (act_b > 0).astype(float)
    intersection = (bin_a * bin_b).sum()
    union = ((bin_a + bin_b) > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def compute_conditional_pearson(act_a, act_b, min_coactive=10):
    """Pearson correlation on co-active samples only."""
    mask = (act_a > 0) & (act_b > 0)
    n_coactive = mask.sum()
    if n_coactive < min_coactive:
        return float('nan'), int(n_coactive)

    vals_a = act_a[mask]
    vals_b = act_b[mask]

    if np.std(vals_a) == 0 or np.std(vals_b) == 0:
        return float('nan'), int(n_coactive)

    corr, _ = pearsonr(vals_a, vals_b)
    return float(corr), int(n_coactive)


def compute_behavioral_similarity(act_matrix_a, feats_a, act_matrix_b, feats_b):
    """
    Compute pairwise behavioral similarity between all features of model A and model B.
    Returns: jaccard_matrix, pearson_matrix, n_coactive_matrix
    """
    n_a = len(feats_a)
    n_b = len(feats_b)

    jaccard = np.zeros((n_a, n_b))
    pearson = np.full((n_a, n_b), np.nan)
    n_coactive = np.zeros((n_a, n_b), dtype=int)

    print(f"Computing {n_a} x {n_b} = {n_a * n_b} behavioral similarity pairs...")

    for i, fa in enumerate(feats_a):
        col_a = act_matrix_a[:, fa]
        for j, fb in enumerate(feats_b):
            col_b = act_matrix_b[:, fb]
            jaccard[i, j] = compute_jaccard(col_a, col_b)
            pearson[i, j], n_coactive[i, j] = compute_conditional_pearson(col_a, col_b)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n_a} rows done")

    return jaccard, pearson, n_coactive


# ======================================================================
# Semantic Similarity
# ======================================================================
def compute_semantic_similarity(descs_a, feats_a, descs_b, feats_b):
    """
    Compute pairwise semantic similarity using SentenceTransformer.
    Returns: cosine_similarity_matrix
    """
    print("Loading SentenceTransformer model...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get description texts in order
    texts_a = [descs_a[str(int(f))]["description"] for f in feats_a]
    texts_b = [descs_b[str(int(f))]["description"] for f in feats_b]

    print(f"Encoding {len(texts_a)} + {len(texts_b)} descriptions...")
    emb_a = st_model.encode(texts_a, normalize_embeddings=True)
    emb_b = st_model.encode(texts_b, normalize_embeddings=True)

    # Cosine similarity (embeddings already normalized)
    cosine_sim = emb_a @ emb_b.T

    return cosine_sim


# ======================================================================
# Feature Matching (Greedy k-NN)
# ======================================================================
def greedy_best_match(sim_matrix, k=1):
    """
    For each feature in model A, find the best-matching feature in model B.
    Returns list of (idx_a, idx_b, similarity) tuples.
    """
    matches = []
    for i in range(sim_matrix.shape[0]):
        best_j = np.argmax(sim_matrix[i])
        matches.append((i, int(best_j), float(sim_matrix[i, best_j])))
    return matches


# ======================================================================
# Money Plot
# ======================================================================
def money_plot(behavioral_scores, semantic_scores, labels_a, labels_b,
               title, save_path, pair_info=None):
    """
    Scatter plot: x = behavioral similarity, y = semantic similarity.
    Each point = one (feature_A, feature_B) pair (best matches only).
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    valid = ~np.isnan(behavioral_scores) & ~np.isnan(semantic_scores)
    beh = behavioral_scores[valid]
    sem = semantic_scores[valid]

    ax.scatter(beh, sem, alpha=0.5, s=30, c='steelblue', edgecolors='none')

    # Add correlation line if enough points
    if len(beh) >= 5:
        corr, pval = pearsonr(beh, sem)
        z = np.polyfit(beh, sem, 1)
        p = np.poly1d(z)
        x_line = np.linspace(beh.min(), beh.max(), 100)
        ax.plot(x_line, p(x_line), '--', color='red', alpha=0.7,
                label=f'r={corr:.3f} (p={pval:.2e})')
        ax.legend(fontsize=11)

    ax.set_xlabel("Behavioral Similarity (Jaccard)", fontsize=13)
    ax.set_ylabel("Semantic Similarity (Cosine)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # Quadrant lines
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    # Quadrant labels
    ax.text(0.75, 0.9, "True Match", ha='center', fontsize=9, color='green', alpha=0.7)
    ax.text(0.25, 0.1, "True Non-Match", ha='center', fontsize=9, color='gray', alpha=0.7)
    ax.text(0.75, 0.1, "Description\nFailure", ha='center', fontsize=9, color='orange', alpha=0.7)
    ax.text(0.25, 0.9, "LLM\nHallucination", ha='center', fontsize=9, color='red', alpha=0.7)

    n_valid = valid.sum()
    n_total = len(behavioral_scores)
    ax.text(0.02, 0.02, f"N={n_valid} valid pairs (of {n_total} best matches)",
            transform=ax.transAxes, fontsize=9, alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Money Plot saved to: {save_path}")

    return corr if len(beh) >= 5 else float('nan')


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--model_b", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--gemma_layer", type=int, default=None,
                        help="Override Gemma best layer (set after probing completes)")
    args = parser.parse_args()

    if args.gemma_layer is not None:
        MODEL_CONFIGS["gemma2"]["layer"] = args.gemma_layer

    model_a_key = args.model_a
    model_b_key = args.model_b
    cfg_a = MODEL_CONFIGS[model_a_key]
    cfg_b = MODEL_CONFIGS[model_b_key]

    if cfg_a["layer"] is None or cfg_b["layer"] is None:
        raise ValueError("Layer not set. Use --gemma_layer for Gemma.")

    print(f"\n{'='*70}")
    print(f"Dual-Alignment: {cfg_a['name']} (L{cfg_a['layer']}) vs {cfg_b['name']} (L{cfg_b['layer']})")
    print(f"{'='*70}\n")

    # Load data
    act_a, feats_a, descs_a = load_model_data(model_a_key)
    act_b, feats_b, descs_b = load_model_data(model_b_key)

    # Step 3: Behavioral similarity
    print(f"\n--- Step 3: Behavioral Similarity ---")
    jaccard, pearson, n_coactive = compute_behavioral_similarity(
        act_a, feats_a, act_b, feats_b
    )

    n_valid_pearson = (~np.isnan(pearson)).sum()
    n_total_pairs = pearson.size
    print(f"  Valid Pearson pairs: {n_valid_pearson}/{n_total_pairs} "
          f"({n_valid_pearson/n_total_pairs:.1%})")
    print(f"  Jaccard: mean={np.mean(jaccard):.4f}, max={np.max(jaccard):.4f}")
    valid_p = pearson[~np.isnan(pearson)]
    if len(valid_p) > 0:
        print(f"  Pearson (valid): mean={np.mean(valid_p):.4f}, max={np.max(valid_p):.4f}")

    # Step 4: Semantic similarity
    print(f"\n--- Step 4: Semantic Similarity ---")
    cosine_sim = compute_semantic_similarity(descs_a, feats_a, descs_b, feats_b)
    print(f"  Cosine similarity: mean={np.mean(cosine_sim):.4f}, max={np.max(cosine_sim):.4f}")

    # Step 5: Feature matching (greedy best-match from A→B)
    print(f"\n--- Step 5: Feature Matching ---")
    # Use combined score for matching: average of Jaccard and cosine
    # (Pearson has too many NaNs for matching)
    combined = (jaccard + cosine_sim) / 2.0
    matches = greedy_best_match(combined)

    # For each match, get behavioral and semantic scores
    match_behavioral = np.array([jaccard[m[0], m[1]] for m in matches])
    match_semantic = np.array([cosine_sim[m[0], m[1]] for m in matches])
    match_pearson = np.array([pearson[m[0], m[1]] for m in matches])

    # Print top matches
    sorted_matches = sorted(matches, key=lambda x: x[2], reverse=True)
    print(f"\nTop 10 matches (combined score):")
    for idx_a, idx_b, score in sorted_matches[:10]:
        fa = int(feats_a[idx_a])
        fb = int(feats_b[idx_b])
        j = jaccard[idx_a, idx_b]
        s = cosine_sim[idx_a, idx_b]
        p = pearson[idx_a, idx_b]
        desc_a = descs_a[str(fa)]["description"][:60]
        desc_b = descs_b[str(fb)]["description"][:60]
        print(f"  {cfg_a['name']} F{fa} <-> {cfg_b['name']} F{fb}")
        print(f"    Jaccard={j:.3f}, Cosine={s:.3f}, Pearson={p:.3f}")
        print(f"    A: {desc_a}...")
        print(f"    B: {desc_b}...")

    # Step 6: Money Plot
    print(f"\n--- Step 6: Money Plot ---")
    pair_label = f"{model_a_key}_vs_{model_b_key}"
    plot_title = f"Dual-Alignment: {cfg_a['name']} vs {cfg_b['name']} (SST-2)"
    plot_path = os.path.join(OUTPUT_DIR, f"money_plot_{pair_label}.png")

    overall_corr = money_plot(
        match_behavioral, match_semantic,
        [int(f) for f in feats_a], [int(f) for f in feats_b],
        plot_title, plot_path
    )
    print(f"  Overall behavioral-semantic correlation: {overall_corr:.4f}")

    # Also make a version using Pearson instead of Jaccard for behavioral
    plot_path_pearson = os.path.join(OUTPUT_DIR, f"money_plot_{pair_label}_pearson.png")
    money_plot(
        match_pearson, match_semantic,
        [int(f) for f in feats_a], [int(f) for f in feats_b],
        f"{plot_title}\n(Behavioral = Cond. Pearson)",
        plot_path_pearson
    )

    # Save full results
    results = {
        "model_a": model_a_key,
        "model_b": model_b_key,
        "model_a_name": cfg_a["name"],
        "model_b_name": cfg_b["name"],
        "layer_a": cfg_a["layer"],
        "layer_b": cfg_b["layer"],
        "n_features_a": len(feats_a),
        "n_features_b": len(feats_b),
        "jaccard_mean": float(np.mean(jaccard)),
        "jaccard_max": float(np.max(jaccard)),
        "cosine_mean": float(np.mean(cosine_sim)),
        "cosine_max": float(np.max(cosine_sim)),
        "pearson_valid_ratio": float(n_valid_pearson / n_total_pairs),
        "overall_correlation": float(overall_corr) if not np.isnan(overall_corr) else None,
        "top_matches": [
            {
                "feature_a": int(feats_a[m[0]]),
                "feature_b": int(feats_b[m[1]]),
                "jaccard": float(jaccard[m[0], m[1]]),
                "cosine": float(cosine_sim[m[0], m[1]]),
                "pearson": float(pearson[m[0], m[1]]) if not np.isnan(pearson[m[0], m[1]]) else None,
                "combined": float(m[2]),
                "desc_a": descs_a[str(int(feats_a[m[0]]))]["description"],
                "desc_b": descs_b[str(int(feats_b[m[1]]))]["description"],
            }
            for m in sorted_matches[:30]
        ],
    }

    results_path = os.path.join(OUTPUT_DIR, f"results_{pair_label}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {results_path}")

    # Save matrices for further analysis
    np.savez(
        os.path.join(OUTPUT_DIR, f"matrices_{pair_label}.npz"),
        jaccard=jaccard,
        pearson=pearson,
        cosine_sim=cosine_sim,
        n_coactive=n_coactive,
        features_a=feats_a,
        features_b=feats_b,
    )
    print(f"Matrices saved to: {OUTPUT_DIR}/matrices_{pair_label}.npz")

    print(f"\n{'='*70}")
    print("DUAL-ALIGNMENT ANALYSIS COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
