# -*- coding: utf-8 -*-
"""Gemma-2B + Gemma Scope Attention SAE (MRPC Paraphrase Detection)

Stage 1: Baseline zero-shot + Layer-wise probing + SAE feature analysis
- Model: google/gemma-2-2b (TransformerLens)
- SAE: gemma-scope-2b-pt-att-canonical (16k features, attention hook)
- Dataset: GLUE MRPC (sentence pairs, paraphrase / non-paraphrase)
"""

# ================================================================
# 0. Imports
# ================================================================
import os
import gc
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from collections import Counter
from huggingface_hub import login

# ================================================================
# 1. Setup
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_att")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# HF auth (for gated Gemma model)
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged in to Hugging Face")

# ================================================================
# 2. Load Gemma-2B and MRPC
# ================================================================
print("Loading Gemma-2-2B base model (for Gemma Scope SAEs)...")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
model.eval()

print(f"Model has {model.cfg.n_layers} layers (0-{model.cfg.n_layers-1})")

print("Loading MRPC dataset...")
dataset = load_dataset("glue", "mrpc")
train_dataset = dataset["train"].select(range(min(2000, len(dataset["train"]))))
val_dataset = dataset["validation"]

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

SEP = "<|endoftext|>"

train_combined = [f"{ex['sentence1']} {SEP} {ex['sentence2']}" for ex in train_dataset]
val_combined = [f"{ex['sentence1']} {SEP} {ex['sentence2']}" for ex in val_dataset]
train_labels_list = list(train_dataset["label"])
val_labels_list = list(val_dataset["label"])

print(f"Combined train: {len(train_combined)}, val: {len(val_combined)}")

# ================================================================
# 3. STEP 1: Baseline - Zero-Shot Prompting
# ================================================================
print("\n" + "="*70)
print("STEP 1: BASELINE - ZERO-SHOT PROMPTING")
print("="*70)

baseline_predictions = []
baseline_labels = []

with torch.no_grad():
    for idx in tqdm(range(len(val_combined)), desc="Zero-Shot Evaluation"):
        sentence = val_combined[idx]
        true_label = val_labels_list[idx]

        prompt = f"""Determine if the following two sentences are paraphrases of each other.

Sentences: {sentence}
Answer (yes or no):"""

        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens)

        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        generated_text = model.tokenizer.decode([next_token_id]).strip().lower()

        if "yes" in generated_text:
            prediction = 1
        elif "no" in generated_text:
            prediction = 0
        else:
            yes_token_id = model.to_tokens(" yes", prepend_bos=False)[0, 0]
            no_token_id = model.to_tokens(" no", prepend_bos=False)[0, 0]
            if next_token_logits[yes_token_id] > next_token_logits[no_token_id]:
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
    f.write("BASELINE - ZERO-SHOT PROMPTING (Gemma-2B Base, MRPC)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n")
    f.write(f"Precision: {baseline_p:.4f}\n")
    f.write(f"Recall: {baseline_r:.4f}\n")
    f.write(f"F1-Score: {baseline_f1:.4f}\n")

print("Baseline saved")

# ================================================================
# 4. STEP 2: Layer-wise Analysis (Probes + Gemma Scope Attention SAEs)
# ================================================================
print("\n" + "="*70)
print("STEP 2: LAYER-WISE ANALYSIS (Gemma Scope Attention SAEs)")
print("="*70)

layer_performance = {}
sae_feature_stats = {}

TOP_K = 10
release = "gemma-scope-2b-pt-att-canonical"

for layer_num in range(model.cfg.n_layers):
    print(f"\n{'='*70}")
    print(f"LAYER {layer_num} ANALYSIS")
    print(f"{'='*70}")

    hook_name = f"blocks.{layer_num}.attn.hook_z"

    # ----------------------------------------------------------
    # Part A: Raw representation classifier (linear probe)
    # ----------------------------------------------------------
    print("\n--- Part A: Raw Representation Classifier ---")

    print("Extracting training representations...")
    train_reps = []
    train_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(train_combined)), desc=f"Train L{layer_num}"):
            sentence = train_combined[idx]
            label = train_labels_list[idx]

            tokens = model.to_tokens(sentence, prepend_bos=True, truncate=True)
            _, cache = model.run_with_cache(tokens)

            layer_acts = cache[hook_name]
            pooled = layer_acts.mean(dim=1)
            pooled_flat = pooled.cpu().numpy().flatten()

            train_reps.append(pooled_flat)
            train_labels.append(label)

    X_train = np.array(train_reps)
    y_train = np.array(train_labels)

    print("Extracting validation representations...")
    val_reps = []
    val_labels_raw = []

    with torch.no_grad():
        for idx in tqdm(range(len(val_combined)), desc=f"Val L{layer_num}"):
            sentence = val_combined[idx]
            label = val_labels_list[idx]

            tokens = model.to_tokens(sentence, prepend_bos=True, truncate=True)
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

    # ----------------------------------------------------------
    # Part B: Gemma Scope SAE feature analysis
    # ----------------------------------------------------------
    print("\n--- Part B: Gemma Scope SAE Feature Analysis ---")

    sae_id = f"layer_{layer_num}/width_16k/canonical"

    try:
        print(f"Loading Gemma Scope SAE: {release}, {sae_id}")
        try:
            sae = SAE.from_pretrained(release, sae_id)
        except:
            sae = SAE.from_pretrained(release, sae_id)[0]

        sae.to(device)
        sae.eval()
        print(f"Loaded SAE: {sae_id}")

    except Exception as e:
        print(f"Could not load SAE for layer {layer_num}: {e}")
        sae_feature_stats[layer_num] = {"error": str(e)}
        continue

    print("Extracting SAE features...")

    pos_active_features = set()
    neg_active_features = set()

    pos_feature_activations = {}
    neg_feature_activations = {}

    pos_feature_counts = Counter()
    neg_feature_counts = Counter()

    total_pos = 0
    total_neg = 0

    with torch.no_grad():
        for idx in tqdm(range(len(val_combined)), desc=f"SAE L{layer_num}"):
            sentence = val_combined[idx]
            label = val_labels_list[idx]

            tokens = model.to_tokens(sentence, prepend_bos=True, truncate=True)
            _, cache = model.run_with_cache(tokens)

            hook_acts = cache[hook_name]
            sae_feature_acts = sae.encode(hook_acts)

            pooled_features = sae_feature_acts.mean(dim=1)
            pooled_features = pooled_features.cpu().numpy().flatten()

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

    common_features = pos_active_features & neg_active_features
    pos_only_features = pos_active_features - neg_active_features
    neg_only_features = neg_active_features - pos_active_features

    pos_avg_activations = {f: np.mean(acts) for f, acts in pos_feature_activations.items()}
    neg_avg_activations = {f: np.mean(acts) for f, acts in neg_feature_activations.items()}

    top5_pos_by_activation = sorted(
        pos_avg_activations.items(), key=lambda x: x[1], reverse=True
    )[:5]
    top5_neg_by_activation = sorted(
        neg_avg_activations.items(), key=lambda x: x[1], reverse=True
    )[:5]

    topk_pos_by_frequency = pos_feature_counts.most_common(TOP_K)
    topk_neg_by_frequency = neg_feature_counts.most_common(TOP_K)

    top5_pos_by_frequency = pos_feature_counts.most_common(5)
    top5_neg_by_frequency = neg_feature_counts.most_common(5)

    topk_pos_ids = {feat for feat, _ in topk_pos_by_frequency}
    topk_neg_ids = {feat for feat, _ in topk_neg_by_frequency}
    topk_common_ids = topk_pos_ids & topk_neg_ids

    sae_feature_stats[layer_num] = {
        "total_pos_features": len(pos_active_features),
        "total_neg_features": len(neg_active_features),
        "common_features": len(common_features),
        "pos_only_features": len(pos_only_features),
        "neg_only_features": len(neg_only_features),
        "top5_pos_by_activation": top5_pos_by_activation,
        "top5_neg_by_activation": top5_neg_by_activation,
        "top5_pos_by_frequency": top5_pos_by_frequency,
        "top5_neg_by_frequency": top5_neg_by_frequency,
        "topk_pos_by_frequency": topk_pos_by_frequency,
        "topk_neg_by_frequency": topk_neg_by_frequency,
        "topk_common_ids": topk_common_ids,
        "total_pos_samples": total_pos,
        "total_neg_samples": total_neg,
    }

    # Save per-layer detailed file
    with open(os.path.join(RESULTS_DIR, f"layer_{layer_num}_complete_analysis.txt"), "w") as f:
        f.write("="*70 + "\n")
        f.write(f"LAYER {layer_num} - COMPLETE ANALYSIS (Gemma Scope Attention SAE, MRPC)\n")
        f.write("="*70 + "\n\n")

        f.write("A. LAYER PERFORMANCE (Raw Representations)\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precision: {p:.4f}\n")
        f.write(f"Recall: {r:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Baseline Accuracy: {baseline_acc:.4f}\n")
        f.write(f"Improvement over baseline: {(acc - baseline_acc)*100:+.2f}%\n\n")

        f.write("Confusion Matrix:\n")
        f.write("              Predicted\n")
        f.write("              Neg    Pos\n")
        f.write(f"Actual Neg  [{cm[0,0]:5d}  {cm[0,1]:5d}]\n")
        f.write(f"       Pos  [{cm[1,0]:5d}  {cm[1,1]:5d}]\n\n")

        f.write("B. SAE FEATURE COUNTS\n")
        f.write("-"*70 + "\n")
        f.write(f"Total PARAPHRASE samples: {total_pos}\n")
        f.write(f"Total NON-PARAPHRASE samples: {total_neg}\n\n")
        f.write(f"Total features activated for PARAPHRASE: {len(pos_active_features)}\n")
        f.write(f"Total features activated for NON-PARAPHRASE: {len(neg_active_features)}\n")
        f.write(f"COMMON features (activated for both): {len(common_features)}\n")
        f.write(f"UNIQUE to PARAPHRASE only: {len(pos_only_features)}\n")
        f.write(f"UNIQUE to NON-PARAPHRASE only: {len(neg_only_features)}\n\n")

        f.write("C. TOP 5 MOST ACTIVATING FEATURES (PARAPHRASE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, avg_act) in enumerate(top5_pos_by_activation, 1):
            freq = pos_feature_counts[feat_idx]
            pct = freq / total_pos * 100
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: avg_activation={avg_act:.4f}, "
                f"frequency={freq}/{total_pos} ({pct:.1f}%) | {status}\n"
            )

        f.write("\nD. TOP 5 MOST ACTIVATING FEATURES (NON-PARAPHRASE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, avg_act) in enumerate(top5_neg_by_activation, 1):
            freq = neg_feature_counts[feat_idx]
            pct = freq / total_neg * 100
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: avg_activation={avg_act:.4f}, "
                f"frequency={freq}/{total_neg} ({pct:.1f}%) | {status}\n"
            )

        f.write("\nE. TOP 5 MOST FREQUENT FEATURES (PARAPHRASE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(top5_pos_by_frequency, 1):
            pct = count / total_pos * 100
            avg_act = pos_avg_activations.get(feat_idx, 0.0)
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%), "
                f"avg_activation={avg_act:.4f} | {status}\n"
            )

        f.write("\nF. TOP 5 MOST FREQUENT FEATURES (NON-PARAPHRASE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(top5_neg_by_frequency, 1):
            pct = count / total_neg * 100
            avg_act = neg_avg_activations.get(feat_idx, 0.0)
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%), "
                f"avg_activation={avg_act:.4f} | {status}\n"
            )

        f.write(f"\nG. TOP {TOP_K} MOST FREQUENT FEATURES (PARAPHRASE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(topk_pos_by_frequency, 1):
            pct = count / total_pos * 100
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%)\n"
            )

        f.write(f"\nH. TOP {TOP_K} MOST FREQUENT FEATURES (NON-PARAPHRASE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(topk_neg_by_frequency, 1):
            pct = count / total_neg * 100
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%)\n"
            )

        f.write(
            f"\nI. COMMON FEATURES AMONG TOP-{TOP_K} PARAPHRASE & NON-PARAPHRASE\n"
        )
        f.write("-"*70 + "\n")
        f.write(f"Count: {len(topk_common_ids)}\n")
        if len(topk_common_ids) > 0:
            f.write(
                "Example IDs: "
                + ", ".join(str(x) for x in sorted(list(topk_common_ids))[:20])
                + "\n"
            )

    print(f"Layer {layer_num} complete analysis saved")

    del sae
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ================================================================
# 5. FINAL SUMMARY
# ================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

best_layer = max(layer_performance.items(), key=lambda x: x[1]["accuracy"])
worst_layer = min(layer_performance.items(), key=lambda x: x[1]["accuracy"])

print(
    f"\n{'Layer':<6} {'Acc%':<8} {'Improv':<10} {'Para Feat':<12} "
    f"{'NonP Feat':<12} {'Common':<10} {'Para Only':<12} {'NonP Only':<12}"
)
print("-"*94)
for layer_num in sorted(layer_performance.keys()):
    perf = layer_performance[layer_num]
    stats = sae_feature_stats.get(layer_num, {})
    print(
        f"L{layer_num:<5} {perf['accuracy']*100:5.2f}%   "
        f"{perf['improvement']*100:+6.2f}%    "
        f"{stats.get('total_pos_features', 'N/A'):<12} "
        f"{stats.get('total_neg_features', 'N/A'):<12} "
        f"{stats.get('common_features', 'N/A'):<10} "
        f"{stats.get('pos_only_features', 'N/A'):<12} "
        f"{stats.get('neg_only_features', 'N/A'):<12}"
    )

print(
    f"\nBest layer: Layer {best_layer[0]} "
    f"({best_layer[1]['accuracy']*100:.2f}% accuracy)"
)
print(
    f"Worst layer: Layer {worst_layer[0]} "
    f"({worst_layer[1]['accuracy']*100:.2f}% accuracy)"
)

with open(os.path.join(RESULTS_DIR, "final_summary.txt"), "w") as f:
    f.write("="*70 + "\n")
    f.write("FINAL SUMMARY - GEMMA SCOPE ATTENTION SAE LAYER-WISE ANALYSIS (MRPC)\n")
    f.write("="*70 + "\n\n")

    f.write("1. BASELINE (Zero-Shot Prompting)\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {baseline_acc*100:.2f}%\n")
    f.write(f"F1-Score: {baseline_f1:.4f}\n\n")

    f.write("2. LAYER-WISE PERFORMANCE & SAE FEATURE STATISTICS\n")
    f.write("-"*70 + "\n")
    f.write(
        f"{'Layer':<6} {'Acc%':<8} {'Improv':<10} {'Para Feat':<12} "
        f"{'NonP Feat':<12} {'Common':<10} {'Para Only':<12} "
        f"{'NonP Only':<12}\n"
    )
    f.write("-"*94 + "\n")
    for layer_num in sorted(layer_performance.keys()):
        perf = layer_performance[layer_num]
        stats = sae_feature_stats.get(layer_num, {})
        f.write(
            f"L{layer_num:<5} {perf['accuracy']*100:5.2f}%   "
            f"{perf['improvement']*100:+6.2f}%    "
            f"{stats.get('total_pos_features', 'N/A'):<12} "
            f"{stats.get('total_neg_features', 'N/A'):<12} "
            f"{stats.get('common_features', 'N/A'):<10} "
            f"{stats.get('pos_only_features', 'N/A'):<12} "
            f"{stats.get('neg_only_features', 'N/A'):<12}\n"
        )

    f.write("\n3. KEY FINDINGS\n")
    f.write("-"*70 + "\n")
    f.write(f"Best Performing Layer: Layer {best_layer[0]}\n")
    f.write(f"  Accuracy: {best_layer[1]['accuracy']*100:.2f}%\n")
    f.write(
        f"  Improvement over baseline: "
        f"{best_layer[1]['improvement']*100:+.2f}%\n\n"
    )
    f.write(f"Worst Performing Layer: Layer {worst_layer[0]}\n")
    f.write(f"  Accuracy: {worst_layer[1]['accuracy']*100:.2f}%\n\n")
    f.write("INTERPRETATION:\n")
    f.write("- Higher accuracy = more paraphrase information in that layer\n")
    f.write("- Many paraphrase-only / non-paraphrase-only SAE features = stronger class separation\n")
    f.write("- Overlap among top-k frequent features quantifies common concepts\n")

print(f"\nAll results saved to {RESULTS_DIR}")
print("="*70)
