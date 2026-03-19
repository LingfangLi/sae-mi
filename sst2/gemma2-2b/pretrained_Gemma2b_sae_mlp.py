"""
Gemma 2 2B SST-2 Analysis with MLP-out SAE (Gemma Scope)
Hook: blocks.N.hook_mlp_out
SAE: gemma-scope-2b-pt-mlp-canonical (16k features)
Model: google/gemma-2-2b (Gemma 2nd gen, d_model=2304, 26 layers)
Pooling: Max Pooling for SAE features
"""

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from collections import Counter
import os

# ======================================================================
# Setup
# ======================================================================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_mlp")
os.makedirs(RESULTS_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ======================================================================
# Load model and data
# ======================================================================
print("Loading Gemma 2 2B base model (for Gemma Scope SAEs)...")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
model.eval()

print(f"Model has {model.cfg.n_layers} layers (0-{model.cfg.n_layers-1})")

print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"].select(range(5000))
val_dataset = dataset["validation"]

print(f"Train samples (subsampled): {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ======================================================================
# STEP 1: Baseline - Zero-Shot Prompting
# ======================================================================
print("\n" + "="*70)
print("STEP 1: BASELINE - ZERO-SHOT PROMPTING")
print("="*70)

baseline_predictions = []
baseline_labels = []

with torch.no_grad():
    for idx in tqdm(range(len(val_dataset)), desc="Zero-Shot Evaluation"):
        sentence = val_dataset[idx]["sentence"]
        true_label = val_dataset[idx]["label"]

        prompt = f"""Classify the sentiment as positive or negative.

Sentence: {sentence}
Sentiment:"""

        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens)

        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        generated_text = model.tokenizer.decode([next_token_id]).strip().lower()

        if "positive" in generated_text or generated_text.startswith("pos"):
            prediction = 1
        elif "negative" in generated_text or generated_text.startswith("neg"):
            prediction = 0
        else:
            pos_token_id = model.to_tokens(" positive", prepend_bos=False)[0, 0]
            neg_token_id = model.to_tokens(" negative", prepend_bos=False)[0, 0]
            if next_token_logits[pos_token_id] > next_token_logits[neg_token_id]:
                prediction = 1
            else:
                prediction = 0

        baseline_predictions.append(prediction)
        baseline_labels.append(true_label)

baseline_predictions = np.array(baseline_predictions)
baseline_labels = np.array(baseline_labels)

baseline_acc = accuracy_score(baseline_labels, baseline_predictions)
baseline_p, baseline_r, baseline_f1, _ = precision_recall_fscore_support(
    baseline_labels, baseline_predictions, average='binary'
)

print(f"\nBaseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
print(f"F1-Score: {baseline_f1:.4f}")

with open(os.path.join(RESULTS_DIR, 'baseline_results.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("BASELINE - ZERO-SHOT PROMPTING (Gemma-2B, MLP-out hook)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n")
    f.write(f"Precision: {baseline_p:.4f}\n")
    f.write(f"Recall: {baseline_r:.4f}\n")
    f.write(f"F1-Score: {baseline_f1:.4f}\n")

# ======================================================================
# STEP 2: Layer-wise Analysis (Probes + Gemma Scope MLP SAEs)
# ======================================================================
print("\n" + "="*70)
print("STEP 2: LAYER-WISE ANALYSIS (Gemma Scope MLP SAEs)")
print("="*70)

layer_performance = {}
sae_feature_stats = {}

TOP_K = 10

SAE_RELEASE = "gemma-scope-2b-pt-mlp-canonical"

for layer_num in range(model.cfg.n_layers):
    print(f"\n{'='*70}")
    print(f"LAYER {layer_num} ANALYSIS (MLP-out)")
    print(f"{'='*70}")

    hook_name = f"blocks.{layer_num}.hook_mlp_out"

    # --------------------------------------------------------------
    # Part A: Raw representation classifier (linear probe on MLP out)
    # --------------------------------------------------------------
    print("\n--- Part A: Raw MLP-out Representation Classifier ---")

    print("Extracting training representations...")
    train_reps = []
    train_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset)), desc=f"Train L{layer_num}"):
            sentence = train_dataset[idx]["sentence"]
            label = train_dataset[idx]["label"]

            tokens = model.to_tokens(sentence, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            layer_acts = cache[hook_name]
            pooled = layer_acts.mean(dim=1)  # mean pooling for raw reps is fine
            pooled_flat = pooled.cpu().numpy().flatten()

            train_reps.append(pooled_flat)
            train_labels.append(label)

    X_train = np.array(train_reps)
    y_train = np.array(train_labels)

    print("Extracting validation representations...")
    val_reps = []
    val_labels_raw = []

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc=f"Val L{layer_num}"):
            sentence = val_dataset[idx]["sentence"]
            label = val_dataset[idx]["label"]

            tokens = model.to_tokens(sentence, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            layer_acts = cache[hook_name]
            pooled = layer_acts.mean(dim=1)
            pooled_flat = pooled.cpu().numpy().flatten()

            val_reps.append(pooled_flat)
            val_labels_raw.append(label)

    X_val = np.array(val_reps)
    y_val = np.array(val_labels_raw)

    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_val)

    acc = accuracy_score(y_val, predictions)
    p, r, f1, _ = precision_recall_fscore_support(y_val, predictions, average='binary')
    cm = confusion_matrix(y_val, predictions)

    layer_performance[layer_num] = {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1_score": f1,
        "improvement": acc - baseline_acc,
        "confusion_matrix": cm,
    }

    print(f"Layer {layer_num} Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Improvement over baseline: {(acc - baseline_acc)*100:+.2f}%")

    # --------------------------------------------------------------
    # Part B: Gemma Scope MLP SAE feature analysis (MAX POOLING)
    # --------------------------------------------------------------
    print("\n--- Part B: Gemma Scope MLP SAE Feature Analysis ---")

    sae_id = f"layer_{layer_num}/width_16k/canonical"

    try:
        print(f"Loading Gemma Scope MLP SAE: {SAE_RELEASE}, {sae_id}")
        sae = SAE.from_pretrained(SAE_RELEASE, sae_id, device=device)
        sae.eval()
        print(f"Loaded SAE: {sae_id}")

    except Exception as e:
        print(f"Could not load SAE for layer {layer_num}: {e}")
        sae_feature_stats[layer_num] = {"error": str(e)}
        continue

    print("Extracting SAE features (max pooling)...")

    pos_active_features = set()
    neg_active_features = set()

    pos_feature_activations = {}
    neg_feature_activations = {}

    pos_feature_counts = Counter()
    neg_feature_counts = Counter()

    total_pos = 0
    total_neg = 0

    # Store full activation matrix for downstream MI / Dual-Alignment
    all_sae_activations = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc=f"SAE L{layer_num}"):
            sentence = val_dataset[idx]["sentence"]
            label = val_dataset[idx]["label"]

            tokens = model.to_tokens(sentence, prepend_bos=True)
            _, cache = model.run_with_cache(tokens)

            hook_acts = cache[hook_name]
            sae_feature_acts = sae.encode(hook_acts)

            # MAX pooling over sequence dimension
            pooled_features = sae_feature_acts.max(dim=1).values
            pooled_features = pooled_features.cpu().numpy().flatten()

            # Save for downstream analysis
            all_sae_activations.append(pooled_features)
            all_labels.append(label)

            active_indices = np.where(pooled_features > 0)[0]
            active_values = pooled_features[active_indices]

            if label == 1:
                total_pos += 1
                for feat_idx, act_val in zip(active_indices, active_values):
                    pos_active_features.add(feat_idx)
                    pos_feature_counts[feat_idx] += 1
                    pos_feature_activations.setdefault(feat_idx, []).append(act_val)
            else:
                total_neg += 1
                for feat_idx, act_val in zip(active_indices, active_values):
                    neg_active_features.add(feat_idx)
                    neg_feature_counts[feat_idx] += 1
                    neg_feature_activations.setdefault(feat_idx, []).append(act_val)

    # Save full activation matrix and labels
    activation_matrix = np.stack(all_sae_activations)  # (n_samples, n_features)
    np.save(os.path.join(RESULTS_DIR, f"layer_{layer_num}_activations.npy"), activation_matrix)
    np.save(os.path.join(RESULTS_DIR, f"layer_{layer_num}_labels.npy"), np.array(all_labels))
    print(f"Saved activation matrix: {activation_matrix.shape}")

    common_features = pos_active_features & neg_active_features
    pos_only_features = pos_active_features - neg_active_features
    neg_only_features = neg_active_features - pos_active_features

    pos_avg_activations = {f: np.mean(acts) for f, acts in pos_feature_activations.items()}
    neg_avg_activations = {f: np.mean(acts) for f, acts in neg_feature_activations.items()}

    topk_pos_by_frequency = pos_feature_counts.most_common(TOP_K)
    topk_neg_by_frequency = neg_feature_counts.most_common(TOP_K)

    topk_pos_ids = {feat for feat, _ in topk_pos_by_frequency}
    topk_neg_ids = {feat for feat, _ in topk_neg_by_frequency}
    topk_common_ids = topk_pos_ids & topk_neg_ids

    sae_feature_stats[layer_num] = {
        "total_pos_features": len(pos_active_features),
        "total_neg_features": len(neg_active_features),
        "common_features": len(common_features),
        "pos_only_features": len(pos_only_features),
        "neg_only_features": len(neg_only_features),
        "topk_pos_by_frequency": topk_pos_by_frequency,
        "topk_neg_by_frequency": topk_neg_by_frequency,
        "topk_common_ids": topk_common_ids,
        "total_pos_samples": total_pos,
        "total_neg_samples": total_neg,
    }

    print(f"\n=== SAE Feature Statistics for Layer {layer_num} (MLP-out, max pooled) ===")
    print(f"Total POSITIVE samples: {total_pos}")
    print(f"Total NEGATIVE samples: {total_neg}")
    print(f"Total features activated for POSITIVE: {len(pos_active_features)}")
    print(f"Total features activated for NEGATIVE: {len(neg_active_features)}")
    print(f"COMMON features (both): {len(common_features)}")

    print(f"\nTop {TOP_K} most frequent POSITIVE features:")
    for feat_idx, cnt in topk_pos_by_frequency:
        print(f"  Feature {feat_idx}: count={cnt}")

    print(f"\nTop {TOP_K} most frequent NEGATIVE features:")
    for feat_idx, cnt in topk_neg_by_frequency:
        print(f"  Feature {feat_idx}: count={cnt}")

    # save per-layer detailed file
    with open(os.path.join(RESULTS_DIR, f"layer_{layer_num}_mlp_analysis.txt"), "w") as f:
        f.write("="*70 + "\n")
        f.write(f"LAYER {layer_num} - MLP-OUT SAE ANALYSIS (Gemma 2 2B)\n")
        f.write(f"Pooling: Max | SAE features: 16k\n")
        f.write("="*70 + "\n\n")

        f.write("A. LAYER PERFORMANCE (MLP-out Linear Probe)\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precision: {p:.4f}\n")
        f.write(f"Recall: {r:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.4f}\n")
        f.write(f"Improvement over baseline: {(acc - baseline_acc)*100:+.2f}%\n\n")

        f.write("B. SAE FEATURE COUNTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total POSITIVE samples: {total_pos}\n")
        f.write(f"Total NEGATIVE samples: {total_neg}\n\n")
        f.write(f"Total features activated for POSITIVE: {len(pos_active_features)}\n")
        f.write(f"Total features activated for NEGATIVE: {len(neg_active_features)}\n")
        f.write(f"COMMON features: {len(common_features)}\n")
        f.write(f"UNIQUE to POSITIVE only: {len(pos_only_features)}\n")
        f.write(f"UNIQUE to NEGATIVE only: {len(neg_only_features)}\n\n")

        f.write(f"C. TOP {TOP_K} MOST FREQUENT FEATURES (POSITIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(topk_pos_by_frequency, 1):
            pct = count / total_pos * 100
            avg_act = pos_avg_activations.get(feat_idx, 0.0)
            f.write(f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%), avg_act={avg_act:.4f}\n")

        f.write(f"\nD. TOP {TOP_K} MOST FREQUENT FEATURES (NEGATIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(topk_neg_by_frequency, 1):
            pct = count / total_neg * 100
            avg_act = neg_avg_activations.get(feat_idx, 0.0)
            f.write(f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%), avg_act={avg_act:.4f}\n")

    print(f"Layer {layer_num} MLP-out analysis saved")

    # Free SAE memory
    del sae
    torch.cuda.empty_cache()

# ======================================================================
# FINAL SUMMARY
# ======================================================================
print("\n" + "="*70)
print("FINAL SUMMARY (Gemma Scope MLP-out SAE, Max Pooling)")
print("="*70)

best_layer = max(layer_performance.items(), key=lambda x: x[1]["accuracy"])
worst_layer = min(layer_performance.items(), key=lambda x: x[1]["accuracy"])

print(
    f"\n{'Layer':<6} {'Acc%':<8} {'Improv':<10} {'Pos Feat':<10} "
    f"{'Neg Feat':<10} {'Common':<10}"
)
print("-"*60)
for layer_num in sorted(layer_performance.keys()):
    perf = layer_performance[layer_num]
    stats = sae_feature_stats.get(layer_num, {})
    print(
        f"L{layer_num:<5} {perf['accuracy']*100:5.2f}%   "
        f"{perf['improvement']*100:+6.2f}%    "
        f"{stats.get('total_pos_features', 'N/A'):<10} "
        f"{stats.get('total_neg_features', 'N/A'):<10} "
        f"{stats.get('common_features', 'N/A'):<10}"
    )

print(f"\nBest layer: Layer {best_layer[0]} ({best_layer[1]['accuracy']*100:.2f}%)")
print(f"Worst layer: Layer {worst_layer[0]} ({worst_layer[1]['accuracy']*100:.2f}%)")

with open(os.path.join(RESULTS_DIR, "final_summary_mlp.txt"), "w") as f:
    f.write("="*70 + "\n")
    f.write("FINAL SUMMARY - Gemma 2 2B MLP-OUT SAE (SST-2)\n")
    f.write("="*70 + "\n\n")
    f.write("SAE: gemma-scope-2b-pt-mlp-canonical\n")
    f.write("Hook: blocks.N.hook_mlp_out\n")
    f.write("Features: 16k per layer | Pooling: Max\n\n")

    f.write("1. BASELINE (Zero-Shot Prompting)\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {baseline_acc*100:.2f}%\n")
    f.write(f"F1-Score: {baseline_f1:.4f}\n\n")

    f.write("2. LAYER-WISE PERFORMANCE\n")
    f.write("-"*70 + "\n")
    f.write(f"{'Layer':<6} {'Acc%':<8} {'Improv':<10} {'Pos Feat':<10} "
            f"{'Neg Feat':<10} {'Common':<10}\n")
    f.write("-"*60 + "\n")
    for layer_num in sorted(layer_performance.keys()):
        perf = layer_performance[layer_num]
        stats = sae_feature_stats.get(layer_num, {})
        f.write(
            f"L{layer_num:<5} {perf['accuracy']*100:5.2f}%   "
            f"{perf['improvement']*100:+6.2f}%    "
            f"{stats.get('total_pos_features', 'N/A'):<10} "
            f"{stats.get('total_neg_features', 'N/A'):<10} "
            f"{stats.get('common_features', 'N/A'):<10}\n"
        )

    f.write(f"\nBest: Layer {best_layer[0]} ({best_layer[1]['accuracy']*100:.2f}%)\n")
    f.write(f"Worst: Layer {worst_layer[0]} ({worst_layer[1]['accuracy']*100:.2f}%)\n")
    f.write(f"\nActivation matrices saved as layer_N_activations.npy\n")
    f.write(f"Labels saved as layer_N_labels.npy\n")

print(f"\nAll results saved to {RESULTS_DIR}")
