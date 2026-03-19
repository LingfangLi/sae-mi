# -*- coding: utf-8 -*-
"""DeepSeek-R1-Distill-Qwen-1.5B + SAE (SST-2)

Adapted from qwen3_0_6b_pretrained.py
Key changes:
- Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (28 layers, hidden_size=1536)
- SAE: EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k (sparsify format)
  - Trained on MLP outputs (hookpoint: layers.N.mlp), d_in=1536, 65k latents, top-k=32
- SAE loading: uses sparsify library (eai-sparsify) instead of sae-lens
- Feature extraction: captures MLP output via forward hooks (matching SAE hookpoint)
"""

# ================================================================
# 0. Imports
# ================================================================
import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from collections import Counter

# ================================================================
# 1. Setup
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================================================================
# 2. Load DeepSeek-R1-Distill-Qwen-1.5B and SST-2
# ================================================================
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading model: {base_model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()

n_layers = model.config.num_hidden_layers
print(f"Model has {n_layers} layers (0-{n_layers-1})")

print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

print(f"Original sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

train_dataset = train_dataset.select(range(min(2000, len(train_dataset))))
val_dataset = val_dataset.select(range(min(872, len(val_dataset))))  # SST-2 val = 872

print(f"Train samples (subsampled): {len(train_dataset)}")
print(f"Validation samples (subsampled): {len(val_dataset)}")

# ================================================================
# Helper: get layer representation (residual stream, for probing)
# ================================================================
def get_layer_rep(text, layer_idx):
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
        )

    hidden_states = out.hidden_states  # len = n_layers+1
    layer_h = hidden_states[layer_idx + 1]  # (1, seq, d_model)
    return layer_h

# ================================================================
# Helper: get MLP output (for SAE encoding)
# The SAE is trained on MLP outputs (hookpoint: layers.N.mlp),
# so we capture the MLP sub-layer output via a forward hook.
# Uses try/finally to ensure hook is always removed.
# ================================================================
def get_mlp_output(text, layer_idx):
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128,
    ).to(device)

    captured = {}

    def hook_fn(module, input, output):
        captured['out'] = output

    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        hook.remove()

    return captured['out']  # (1, seq, hidden_size=1536)

# ================================================================
# 3. STEP 1: Baseline - Zero-Shot Prompting
# ================================================================
print("\n" + "="*70)
print("STEP 1: BASELINE - ZERO-SHOT PROMPTING (DeepSeek-R1-Distill-Qwen-1.5B)")
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

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=256,
        ).to(device)

        out = model(**enc)
        logits = out.logits  # (1, seq, vocab)
        next_token_logits = logits[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()
        generated_text = tokenizer.decode([next_token_id]).strip().lower()

        if "positive" in generated_text or generated_text.startswith("pos"):
            prediction = 1
        elif "negative" in generated_text or generated_text.startswith("neg"):
            prediction = 0
        else:
            pos_id = tokenizer.encode(" positive")[-1]
            neg_id = tokenizer.encode(" negative")[-1]
            if next_token_logits[pos_id] > next_token_logits[neg_id]:
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

with open(os.path.join(RESULTS_DIR, 'baseline_results_deepseek.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("BASELINE - ZERO-SHOT PROMPTING (DeepSeek-R1-Distill-Qwen-1.5B)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n")
    f.write(f"Precision: {baseline_p:.4f}\n")
    f.write(f"Recall: {baseline_r:.4f}\n")
    f.write(f"F1-Score: {baseline_f1:.4f}\n")

print("Baseline saved")

# ================================================================
# 4. STEP 2: Layer-wise Analysis (Probes + SAE on MLP outputs)
# ================================================================
print("\n" + "="*70)
print("STEP 2: LAYER-WISE ANALYSIS (DeepSeek-R1-Distill-Qwen-1.5B + SAE)")
print("="*70)

layer_performance = {}
sae_feature_stats = {}

TOP_K = 10
sae_repo = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"

for layer_num in range(n_layers):
    print(f"\n{'='*70}")
    print(f"LAYER {layer_num} ANALYSIS")
    print(f"{'='*70}")

    # ----------------------------------------------------------
    # Part A: Raw representation classifier
    # ----------------------------------------------------------
    print("\n--- Part A: Raw Representation Classifier (Hidden state) ---")

    print("Extracting training representations...")
    train_reps = []
    train_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset)), desc=f"Train L{layer_num}"):
            sentence = train_dataset[idx]["sentence"]
            label = train_dataset[idx]["label"]

            layer_h = get_layer_rep(sentence, layer_num)   # (1, seq, d_model)
            pooled = layer_h.mean(dim=1)                  # (1, d_model)
            pooled_flat = pooled[0].to(torch.float32).cpu().numpy().flatten()

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

            layer_h = get_layer_rep(sentence, layer_num)
            pooled = layer_h.mean(dim=1)
            pooled_flat = pooled[0].to(torch.float32).cpu().numpy().flatten()

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
    # Part B: SAE feature analysis (MLP output -> sparsify SAE)
    # ----------------------------------------------------------
    print("\n--- Part B: SAE Feature Analysis (MLP output SAE) ---")

    hookpoint = f"layers.{layer_num}.mlp"
    try:
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
        sae.to(device)
        sae.eval()
        print(f"Loaded SAE: {hookpoint} (d_in={sae.d_in}, num_latents={sae.num_latents})")
    except Exception as e:
        print(f"Could not load SAE for layer {layer_num}: {e}")
        sae_feature_stats[layer_num] = {}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
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

    num_latents = sae.num_latents

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc=f"SAE L{layer_num}"):
            sentence = val_dataset[idx]["sentence"]
            label = val_dataset[idx]["label"]

            # Get MLP output (matching SAE training hookpoint)
            mlp_out = get_mlp_output(sentence, layer_num)  # (1, seq, 1536)
            seq_len = mlp_out.shape[1]

            # Flatten batch+seq for SAE encoding: (seq, 1536)
            flat = mlp_out.view(-1, mlp_out.shape[-1])

            # Encode with sparsify SAE -> sparse TopK output
            enc_output = sae.encode(flat)
            # enc_output.top_acts: (seq, k=32)
            # enc_output.top_indices: (seq, k=32)

            # Convert sparse to dense: (seq, num_latents)
            dense = torch.zeros(seq_len, num_latents, device=device, dtype=torch.float32)
            dense.scatter_(1, enc_output.top_indices.long(), enc_output.top_acts.float())

            # Mean pool over sequence dimension: (num_latents,)
            pooled_features = dense.mean(dim=0).cpu().numpy()

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

    # ----------------------------------------------------------
    # Save per-layer detailed analysis
    # ----------------------------------------------------------
    with open(os.path.join(RESULTS_DIR, f"layer_{layer_num}_complete_analysis_deepseek.txt"), "w") as f:
        f.write("="*70 + "\n")
        f.write(f"LAYER {layer_num} - COMPLETE ANALYSIS (DeepSeek-R1-Distill-Qwen-1.5B)\n")
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
        f.write(f"Total POSITIVE samples: {total_pos}\n")
        f.write(f"Total NEGATIVE samples: {total_neg}\n\n")
        f.write(f"Total features activated for POSITIVE: {len(pos_active_features)}\n")
        f.write(f"Total features activated for NEGATIVE: {len(neg_active_features)}\n")
        f.write(f"COMMON features (activated for both): {len(common_features)}\n")
        f.write(f"UNIQUE to POSITIVE only: {len(pos_only_features)}\n")
        f.write(f"UNIQUE to NEGATIVE only: {len(neg_only_features)}\n\n")

        f.write("C. TOP 5 MOST ACTIVATING FEATURES (POSITIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, avg_act) in enumerate(top5_pos_by_activation, 1):
            freq = pos_feature_counts[feat_idx]
            pct = freq / total_pos * 100 if total_pos > 0 else 0.0
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: avg_activation={avg_act:.4f}, "
                f"frequency={freq}/{total_pos} ({pct:.1f}%) | {status}\n"
            )

        f.write("\nD. TOP 5 MOST ACTIVATING FEATURES (NEGATIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, avg_act) in enumerate(top5_neg_by_activation, 1):
            freq = neg_feature_counts[feat_idx]
            pct = freq / total_neg * 100 if total_neg > 0 else 0.0
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: avg_activation={avg_act:.4f}, "
                f"frequency={freq}/{total_neg} ({pct:.1f}%) | {status}\n"
            )

        f.write("\nE. TOP 5 MOST FREQUENT FEATURES (POSITIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(top5_pos_by_frequency, 1):
            pct = count / total_pos * 100 if total_pos > 0 else 0.0
            avg_act = pos_avg_activations.get(feat_idx, 0.0)
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%), "
                f"avg_activation={avg_act:.4f} | {status}\n"
            )

        f.write("\nF. TOP 5 MOST FREQUENT FEATURES (NEGATIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(top5_neg_by_frequency, 1):
            pct = count / total_neg * 100 if total_neg > 0 else 0.0
            avg_act = neg_avg_activations.get(feat_idx, 0.0)
            status = "COMMON" if feat_idx in common_features else "UNIQUE"
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%), "
                f"avg_activation={avg_act:.4f} | {status}\n"
            )

        f.write(f"\nG. TOP {TOP_K} MOST FREQUENT FEATURES (POSITIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(topk_pos_by_frequency, 1):
            pct = count / total_pos * 100 if total_pos > 0 else 0.0
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%)\n"
            )

        f.write(f"\nH. TOP {TOP_K} MOST FREQUENT FEATURES (NEGATIVE)\n")
        f.write("-"*70 + "\n")
        for rank, (feat_idx, count) in enumerate(topk_neg_by_frequency, 1):
            pct = count / total_neg * 100 if total_neg > 0 else 0.0
            f.write(
                f"{rank}. Feature {feat_idx}: count={count} ({pct:.1f}%)\n"
            )

        f.write(
            f"\nI. COMMON FEATURES AMONG TOP-{TOP_K} POS & NEG\n"
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

    # Free SAE memory before next layer
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
    f"\n{'Layer':<6} {'Acc%':<8} {'Improv':<10} "
    f"{'Pos Feat':<10} {'Neg Feat':<10} {'Common':<10}"
)
print("-"*90)
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

print(
    f"\nBest layer: Layer {best_layer[0]} "
    f"({best_layer[1]['accuracy']*100:.2f}% accuracy)"
)
print(
    f"Worst layer: Layer {worst_layer[0]} "
    f"({worst_layer[1]['accuracy']*100:.2f}% accuracy)"
)

with open(os.path.join(RESULTS_DIR, "final_summary_deepseek.txt"), "w") as f:
    f.write("="*70 + "\n")
    f.write("FINAL SUMMARY - LAYER-WISE ANALYSIS (DeepSeek-R1-Distill-Qwen-1.5B + SAE)\n")
    f.write("="*70 + "\n\n")

    f.write("1. BASELINE (Zero-Shot Prompting)\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {baseline_acc*100:.2f}%\n")
    f.write(f"F1-Score: {baseline_f1:.4f}\n\n")

    f.write("2. LAYER-WISE PERFORMANCE & SAE FEATURE STATISTICS\n")
    f.write("-"*70 + "\n")
    f.write(
        f"{'Layer':<6} {'Acc%':<8} {'Improv':<10} "
        f"{'Pos Feat':<10} {'Neg Feat':<10} {'Common':<10}\n"
    )
    f.write("-"*90 + "\n")
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
    f.write("- Higher accuracy = more sentiment information in that layer\n")
    f.write("- Many pos-only / neg-only SAE features = stronger class separation\n")
    f.write("- Overlap among top-k frequent features quantifies common concepts\n")

print(f"\nAll results saved to {RESULTS_DIR}")
print("="*70)
