# -*- coding: utf-8 -*-
"""Qwen3-8B-Base + Qwen-Scope Residual SAE (Machine Translation - Europarl EN-FR)

Stage 1: Zero-shot baseline + Layer-wise probing + SAE feature analysis (max-pool).
Same structure as mrpc/qwen3-8b/qwen3_8b_mrpc.py, differences:
  - Dataset: Europarl v7 EN-FR (1000 real + 1000 shuffled train, 500+500 val)
  - Input: f"{english} {EOS} {french}"
  - Baseline prompt: correct translation yes/no
Model:  Qwen/Qwen3-8B-Base
SAE:    qwen-scope-3-8b-base-w64k-l100  (residual, d_sae=65536, sae_id=layer{N})
"""

import os
import gc
import json
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ================================================================
# 0. CLI
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, default="all",
                    help="'all' or comma-separated ints")
parser.add_argument("--n_train_real", type=int, default=1000,
                    help="Number of real EN-FR pairs in training (matched by n_train_neg)")
parser.add_argument("--n_val_real", type=int, default=500,
                    help="Number of real EN-FR pairs in val (matched by shuffled negatives)")
parser.add_argument("--skip_sae", action="store_true")
args = parser.parse_args()

# ================================================================
# 1. Paths & env
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault('HF_HOME', CACHE_DIR)
os.environ.setdefault('TRANSFORMERS_CACHE', CACHE_DIR)
os.environ.setdefault('HF_DATASETS_CACHE', CACHE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

# ================================================================
# 2. Load Qwen3-8B-Base
# ================================================================
BASE_MODEL_ID = "Qwen/Qwen3-8B-Base"
SAE_RELEASE = "qwen-scope-3-8b-base-w64k-l100"

print(f"\nLoading model: {BASE_MODEL_ID} (bf16) ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16,
    device_map=device, cache_dir=CACHE_DIR,
)
model.eval()
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model has {n_layers} layers (0-{n_layers-1}), d_model={d_model}")

# ================================================================
# 3. Load Europarl and build balanced train/val
# ================================================================
DATA_DIR = "/mnt/scratch/users/yangwr/Lingfang/saes-mi/data"
EN_PATH = os.path.join(DATA_DIR, "europarl-v7.fr-en.en")
FR_PATH = os.path.join(DATA_DIR, "europarl-v7.fr-en.fr")

print(f"\nLoading Europarl v7 EN-FR from {DATA_DIR} ...")
with open(EN_PATH, 'r', encoding='utf-8') as fe, open(FR_PATH, 'r', encoding='utf-8') as ff:
    en_lines = [l.strip() for l in fe if l.strip()]
    fr_lines = [l.strip() for l in ff if l.strip()]

pairs = list(zip(en_lines[:30000], fr_lines[:30000]))
print(f"Loaded {len(pairs)} aligned pairs (took first 30k)")

random.seed(42)
pos_train = random.sample(pairs, args.n_train_real)
remaining = [p for p in pairs if p not in set(pos_train)]
pos_val = random.sample(remaining, args.n_val_real)

neg_train, neg_val = [], []
used = set()
while len(neg_train) < args.n_train_real:
    en, _ = random.choice(pairs)
    _, wrong_fr = random.choice(pairs)
    p = (en, wrong_fr)
    if p not in used:
        used.add(p)
        neg_train.append(p)

while len(neg_val) < args.n_val_real:
    en, _ = random.choice(pairs)
    _, wrong_fr = random.choice(pairs)
    p = (en, wrong_fr)
    if p not in used:
        used.add(p)
        neg_val.append(p)

train_dataset = [(f"{en} {EOS} {fr}", 1) for en, fr in pos_train] + \
                [(f"{en} {EOS} {fr}", 0) for en, fr in neg_train]
val_dataset = [(f"{en} {EOS} {fr}", 1) for en, fr in pos_val] + \
              [(f"{en} {EOS} {fr}", 0) for en, fr in neg_val]

random.shuffle(train_dataset)
random.shuffle(val_dataset)

train_texts = [t for t, _ in train_dataset]
train_labels_list = [l for _, l in train_dataset]
val_texts = [t for t, _ in val_dataset]
val_labels_list = [l for _, l in val_dataset]
print(f"Train pairs: {len(train_texts)} ({args.n_train_real} real + {args.n_train_real} shuffled)")
print(f"Val   pairs: {len(val_texts)} ({args.n_val_real} real + {args.n_val_real} shuffled)")


@torch.no_grad()
def get_layer_rep(text, layer_idx):
    enc = tokenizer(text, return_tensors="pt", padding=False,
                    truncation=True, max_length=128).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx + 1]


# ================================================================
# 4. STEP 1: Zero-shot baseline (yes/no translation)
# ================================================================
print("\n" + "=" * 70)
print("STEP 1: BASELINE — ZERO-SHOT PROMPTING (Qwen3-8B-Base, MT)")
print("=" * 70)

baseline_predictions, baseline_labels = [], []

with torch.no_grad():
    for idx in tqdm(range(len(val_texts)), desc="Zero-shot"):
        pair = val_texts[idx]
        true_label = val_labels_list[idx]

        prompt = f"""Determine if the following English-French pair is a correct translation.

Sentence pair: {pair}
Answer (yes or no):"""

        enc = tokenizer(prompt, return_tensors="pt", padding=False,
                        truncation=True, max_length=256).to(device)
        logits = model(**enc, use_cache=False).logits
        next_logits = logits[0, -1, :]

        yes_id = tokenizer.encode(" yes", add_special_tokens=False)[-1]
        no_id = tokenizer.encode(" no", add_special_tokens=False)[-1]

        top_tok = tokenizer.decode([int(torch.argmax(next_logits))]).strip().lower()
        if "yes" in top_tok:
            pred = 1
        elif "no" in top_tok:
            pred = 0
        else:
            pred = 1 if next_logits[yes_id] > next_logits[no_id] else 0

        baseline_predictions.append(pred)
        baseline_labels.append(true_label)

baseline_predictions = np.array(baseline_predictions)
baseline_labels = np.array(baseline_labels)

baseline_acc = accuracy_score(baseline_labels, baseline_predictions)
b_p, b_r, b_f1, _ = precision_recall_fscore_support(
    baseline_labels, baseline_predictions, average='binary'
)

print(f"\nBaseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%) | F1: {b_f1:.4f}")

with open(os.path.join(RESULTS_DIR, "baseline_results.txt"), "w") as f:
    f.write("=" * 70 + "\n")
    f.write(f"BASELINE — MT (Europarl) ZERO-SHOT ({BASE_MODEL_ID})\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Accuracy:  {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n")
    f.write(f"Precision: {b_p:.4f}\n")
    f.write(f"Recall:    {b_r:.4f}\n")
    f.write(f"F1:        {b_f1:.4f}\n")
    f.write(f"Val N:     {len(baseline_labels)}\n")

# ================================================================
# 5. STEP 2: Layer-wise probing (cached) + SAE feature stats
# ================================================================
if args.layers == "all":
    layers_to_run = list(range(n_layers))
else:
    layers_to_run = [int(x) for x in args.layers.split(",")]

print("\n" + "=" * 70)
print(f"STEP 2: LAYER-WISE ANALYSIS (layers = {layers_to_run})")
print(f"SAE release = {SAE_RELEASE}  (skip_sae={args.skip_sae})")
print("=" * 70)

layer_performance = {}
sae_feature_stats = {}


@torch.no_grad()
def cache_pooled_reps(texts, labels, name):
    X = np.zeros((len(texts), n_layers, d_model), dtype=np.float32)
    y = np.zeros(len(texts), dtype=np.int64)
    for idx in tqdm(range(len(texts)), desc=f"Cache {name}"):
        enc = tokenizer(texts[idx], return_tensors="pt", padding=False,
                        truncation=True, max_length=128).to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        for L in range(n_layers):
            X[idx, L] = out.hidden_states[L + 1].mean(dim=1)[0].to(torch.float32).cpu().numpy()
        y[idx] = labels[idx]
    return X, y


print("\n--- Part A precompute: cache mean-pooled residual for ALL layers ---")
X_train_all, y_train = cache_pooled_reps(train_texts, train_labels_list, "train")
X_val_all, y_val = cache_pooled_reps(val_texts, val_labels_list, "val")
print(f"Cached  train {X_train_all.shape}   val {X_val_all.shape}")

for layer_num in layers_to_run:
    print(f"\n{'='*70}\nLAYER {layer_num}\n{'='*70}")

    X_train = X_train_all[:, layer_num, :]
    X_val = X_val_all[:, layer_num, :]

    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)

    acc = accuracy_score(y_val, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_val, preds, average='binary')
    cm = confusion_matrix(y_val, preds)

    layer_performance[layer_num] = {
        "accuracy": float(acc), "precision": float(p),
        "recall": float(r), "f1": float(f1),
        "improvement_over_baseline": float(acc - baseline_acc),
        "confusion_matrix": cm.tolist(),
    }

    print(f"L{layer_num} probe acc = {acc:.4f} ({acc*100:.2f}%)  "
          f"Δbaseline = {(acc-baseline_acc)*100:+.2f}%")

    if args.skip_sae:
        continue

    print("--- Part B: SAE feature analysis (Qwen-Scope residual, max-pool) ---")
    sae_id = f"layer{layer_num}"
    try:
        loaded = SAE.from_pretrained(SAE_RELEASE, sae_id)
        sae = loaded[0] if isinstance(loaded, tuple) else loaded
        sae.to(device)
        sae.eval()
        print(f"Loaded SAE {sae_id}: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    except Exception as e:
        print(f"[skip] SAE layer {layer_num} failed to load: {e}")
        sae_feature_stats[layer_num] = {}
        gc.collect()
        torch.cuda.empty_cache()
        continue

    pos_active, neg_active = set(), set()
    pos_counts, neg_counts = Counter(), Counter()
    pos_act_sum, neg_act_sum = {}, {}
    n_pos = n_neg = 0

    with torch.no_grad():
        for idx in tqdm(range(len(val_texts)), desc=f"SAE   L{layer_num}"):
            pair = val_texts[idx]
            label = val_labels_list[idx]

            layer_h = get_layer_rep(pair, layer_num)
            sae_feats = sae.encode(layer_h.to(sae.dtype))
            # MAX-pool over sequence — paper §3.2
            pooled = sae_feats.max(dim=1).values[0].to(torch.float32).cpu().numpy()

            active_idx = np.where(pooled > 0)[0]
            active_val = pooled[active_idx]

            if label == 1:
                n_pos += 1
                for fi, av in zip(active_idx, active_val):
                    pos_active.add(int(fi))
                    pos_counts[int(fi)] += 1
                    pos_act_sum[int(fi)] = pos_act_sum.get(int(fi), 0.0) + float(av)
            else:
                n_neg += 1
                for fi, av in zip(active_idx, active_val):
                    neg_active.add(int(fi))
                    neg_counts[int(fi)] += 1
                    neg_act_sum[int(fi)] = neg_act_sum.get(int(fi), 0.0) + float(av)

    common = pos_active & neg_active
    pos_only = pos_active - neg_active
    neg_only = neg_active - pos_active

    pos_avg = {f: pos_act_sum[f] / pos_counts[f] for f in pos_active}
    neg_avg = {f: neg_act_sum[f] / neg_counts[f] for f in neg_active}

    top5_pos = sorted(pos_avg.items(), key=lambda x: x[1], reverse=True)[:5]
    top5_neg = sorted(neg_avg.items(), key=lambda x: x[1], reverse=True)[:5]

    sae_feature_stats[layer_num] = {
        "n_pos_samples": n_pos, "n_neg_samples": n_neg,
        "n_pos_active": len(pos_active), "n_neg_active": len(neg_active),
        "n_common": len(common), "n_pos_only": len(pos_only), "n_neg_only": len(neg_only),
        "top5_pos_by_avg_activation": [(int(f), float(v)) for f, v in top5_pos],
        "top5_neg_by_avg_activation": [(int(f), float(v)) for f, v in top5_neg],
        "d_sae": int(sae.cfg.d_sae),
        "pct_active_pos": len(pos_active) / int(sae.cfg.d_sae),
        "pct_active_neg": len(neg_active) / int(sae.cfg.d_sae),
    }

    with open(os.path.join(RESULTS_DIR, f"layer_{layer_num}_complete_analysis.txt"), "w") as f:
        f.write(f"LAYER {layer_num} — {BASE_MODEL_ID} + {SAE_RELEASE}/{sae_id} (MT Europarl)\n")
        f.write("=" * 70 + "\n\n")
        f.write("A. LR PROBE ON RAW RESIDUAL\n")
        for k, v in layer_performance[layer_num].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nB. SAE FEATURE COUNTS (max-pool)\n")
        for k, v in sae_feature_stats[layer_num].items():
            f.write(f"  {k}: {v}\n")

    np.savez(
        os.path.join(RESULTS_DIR, f"layer_{layer_num}_active_features.npz"),
        pos_active=np.array(sorted(pos_active), dtype=np.int32),
        neg_active=np.array(sorted(neg_active), dtype=np.int32),
        common=np.array(sorted(common), dtype=np.int32),
        pos_only=np.array(sorted(pos_only), dtype=np.int32),
        neg_only=np.array(sorted(neg_only), dtype=np.int32),
        n_pos_samples=n_pos, n_neg_samples=n_neg,
        d_sae=int(sae.cfg.d_sae),
    )

    del sae
    gc.collect()
    torch.cuda.empty_cache()

# ================================================================
# 6. Final summary
# ================================================================
summary = {
    "task": "MT_Europarl",
    "model": BASE_MODEL_ID,
    "sae_release": SAE_RELEASE,
    "n_layers_total": n_layers,
    "layers_run": layers_to_run,
    "baseline_accuracy": float(baseline_acc),
    "layer_performance": layer_performance,
    "sae_feature_stats": sae_feature_stats,
}

# Full 36-layer run → final_summary.{json,txt}
# Subset run (SAE Part B on best layers only) → sae_partB_summary.{json,txt}
# This prevents SAE follow-up runs from overwriting the full-probing summary.
is_full_run = (len(layers_to_run) == n_layers
               and set(layers_to_run) == set(range(n_layers)))
summary_stem = "final_summary" if is_full_run else "sae_partB_summary"
print(f"\nWriting summary to {summary_stem}.{{json,txt}}  (is_full_run={is_full_run})")

with open(os.path.join(RESULTS_DIR, f"{summary_stem}.json"), "w") as f:
    json.dump(summary, f, indent=2)

with open(os.path.join(RESULTS_DIR, f"{summary_stem}.txt"), "w") as f:
    f.write("=" * 70 + "\n")
    f.write(f"FINAL SUMMARY — MT (Europarl) — {BASE_MODEL_ID} + {SAE_RELEASE}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Baseline zero-shot: {baseline_acc*100:.2f}%\n\n")
    f.write("LR-probe accuracy by layer:\n")
    for L in sorted(layer_performance):
        f.write(f"  L{L:2d}: {layer_performance[L]['accuracy']*100:6.2f}%  "
                f"(Δ={layer_performance[L]['improvement_over_baseline']*100:+.2f}%)\n")

    if sae_feature_stats and any(sae_feature_stats.values()):
        f.write("\nSAE feature counts by layer (pct_active_pos, pct_active_neg, common):\n")
        for L in sorted(sae_feature_stats):
            s = sae_feature_stats[L]
            if not s:
                continue
            f.write(f"  L{L:2d}: pos={s['n_pos_active']:>6} ({s['pct_active_pos']*100:5.2f}%)  "
                    f"neg={s['n_neg_active']:>6} ({s['pct_active_neg']*100:5.2f}%)  "
                    f"common={s['n_common']:>6}\n")

print(f"\nDone. Results written to {RESULTS_DIR}")
