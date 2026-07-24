# -*- coding: utf-8 -*-
"""Firing pattern visualization for Qwen3-8B-Base on MT (Europarl EN-FR translation quality).

Two sample sets split by the model's zero-shot prediction (not GT):
  - Predicted-YES  → model thinks pair is a valid translation
  - Predicted-NO   → model thinks pair is NOT a valid translation

Identify top-10 features that fire more on predicted-YES vs predicted-NO;
plot avg firing on 30 held-out pairs per set.

Model:  Qwen/Qwen3-8B-Base
SAE:    qwen-scope-3-8b-base-w64k-l100 (residual, d_sae=65536)
Layer:  best from Stage-1 probing (default 20 = MT peak 54.50%)
Pool:   MAX (paper §3.2)

Data:   Europarl v7 EN-FR, balanced pool (real pairs + shuffled negatives)
        Same random seed as qwen3_8b_mt.py for consistency with probing.
"""

import os, random, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, default=20,
                    help="Best MT layer for Qwen3-8B (default 20 = peak 54.50%)")
parser.add_argument("--n_select", type=int, default=200)
parser.add_argument("--n_plot", type=int, default=30)
parser.add_argument("--pool_size", type=int, default=3000,
                    help="Candidate pool size (real+shuffled combined)")
args = parser.parse_args()

LAYER = args.layer
N_SELECT = args.n_select
N_PLOT = args.n_plot
MODEL_NAME = f"Qwen3-8B (MT, L{LAYER})"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "qwen3_8b_mt")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', CACHE_DIR)
os.environ.setdefault('TRANSFORMERS_CACHE', CACHE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load Europarl and build balanced candidate pool ───────────
DATA_DIR = "/mnt/scratch/users/yangwr/Lingfang/saes-mi/data"
EN_PATH = os.path.join(DATA_DIR, "europarl-v7.fr-en.en")
FR_PATH = os.path.join(DATA_DIR, "europarl-v7.fr-en.fr")

print(f"Loading Europarl EN-FR ...")
with open(EN_PATH, 'r', encoding='utf-8') as fe, open(FR_PATH, 'r', encoding='utf-8') as ff:
    en_lines = [l.strip() for l in fe if l.strip()]
    fr_lines = [l.strip() for l in ff if l.strip()]
pairs = list(zip(en_lines[:30000], fr_lines[:30000]))
print(f"Loaded {len(pairs)} aligned real pairs (took first 30k)")

random.seed(42)
half = args.pool_size // 2
real_pool = random.sample(pairs, half)
shuffled_pool, used = [], set()
while len(shuffled_pool) < half:
    en, _ = random.choice(pairs)
    _, wrong_fr = random.choice(pairs)
    p = (en, wrong_fr)
    if p not in used:
        used.add(p)
        shuffled_pool.append(p)

# all_pairs: (english, french, gt_label)  gt_label 1 = real, 0 = shuffled
all_pairs = [(en, fr, 1) for en, fr in real_pool] + \
            [(en, fr, 0) for en, fr in shuffled_pool]
random.shuffle(all_pairs)
print(f"Candidate pool: {len(all_pairs)} pairs ({half} real + {half} shuffled)")

# ── Load Model + SAE ─────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

base_model_id = "Qwen/Qwen3-8B-Base"
print(f"Loading {base_model_id} (bf16)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
)
model.eval()

release = "qwen-scope-3-8b-base-w64k-l100"
sae_id = f"layer{LAYER}"
print(f"Loading SAE: {release}, {sae_id}")
loaded = SAE.from_pretrained(release, sae_id)
sae = loaded[0] if isinstance(loaded, tuple) else loaded
sae.to(device).eval()
d_sae = sae.cfg.d_sae
print(f"SAE: layer={LAYER}, d_sae={d_sae}, k={getattr(sae.cfg,'k','?')}")


def combine_pair(en, fr):
    return f"{en} {EOS} {fr}"


def get_layer_rep(text, layer_idx):
    enc = tokenizer(text, return_tensors="pt", padding=False,
                    truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx + 1]


def predict_translation_zeroshot(en, fr):
    # Same prompt as qwen3_8b_mt.py
    pair = combine_pair(en, fr)
    prompt = f"""Determine if the following English-French pair is a correct translation.

Sentence pair: {pair}
Answer (yes or no):"""
    enc = tokenizer(prompt, return_tensors="pt", padding=False,
                    truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    next_logits = out.logits[0, -1, :]
    yes_id = tokenizer.encode(" yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode(" no", add_special_tokens=False)[-1]
    top_tok = tokenizer.decode([int(torch.argmax(next_logits))]).strip().lower()
    if "yes" in top_tok:
        return 1
    if "no" in top_tok:
        return 0
    return 1 if next_logits[yes_id] > next_logits[no_id] else 0


# ── Filter by zero-shot prediction ────────────────────────────
print("\nStep 0: Filtering pairs by zero-shot prediction (by model)...")
yes_select, yes_plot = [], []
no_select, no_plot = [], []
yes_plot_done = no_plot_done = False
for i, (en, fr, gt) in enumerate(all_pairs):
    if yes_plot_done and no_plot_done \
       and len(yes_select) >= N_SELECT and len(no_select) >= N_SELECT:
        break
    pred = predict_translation_zeroshot(en, fr)
    if pred == 1:
        if len(yes_select) < N_SELECT:
            yes_select.append((en, fr, gt))
        elif not yes_plot_done and len(yes_plot) < N_PLOT:
            yes_plot.append((en, fr, gt))
            if len(yes_plot) >= N_PLOT:
                yes_plot_done = True
    else:
        if len(no_select) < N_SELECT:
            no_select.append((en, fr, gt))
        elif not no_plot_done and len(no_plot) < N_PLOT:
            no_plot.append((en, fr, gt))
            if len(no_plot) >= N_PLOT:
                no_plot_done = True
    if (i + 1) % 100 == 0:
        print(f"  Checked {i+1}/{len(all_pairs)}: "
              f"yes_sel={len(yes_select)}, yes_plt={len(yes_plot)}, "
              f"no_sel={len(no_select)}, no_plt={len(no_plot)}")

for name, sel, plt_lst in [("yes", yes_select, yes_plot),
                            ("no", no_select, no_plot)]:
    if len(plt_lst) < N_PLOT:
        extra = N_PLOT - len(plt_lst)
        plt_lst.extend(sel[:extra])
        print(f"  Warning: borrowing {extra} from {name}_select for {name}_plot")

print(f"\nAfter filtering: YES sel={len(yes_select)} plt={len(yes_plot)} | "
      f"NO sel={len(no_select)} plt={len(no_plot)}")


# ── Extract SAE activations (MAX-POOL) ────────────────────────
def get_sae_activation(en, fr):
    text = combine_pair(en, fr)
    layer_h = get_layer_rep(text, LAYER)
    sae_feats = sae.encode(layer_h.to(sae.dtype))
    pooled = sae_feats.max(dim=1).values[0].to(torch.float32).detach().cpu().numpy()
    return pooled


print(f"\nStep 1: Extracting activations ({len(yes_select)} yes, {len(no_select)} no)...")
yes_acts = np.array([get_sae_activation(en, fr) for en, fr, _ in yes_select])
no_acts = np.array([get_sae_activation(en, fr) for en, fr, _ in no_select])

# ── Feature selection ────────────────────────────────────────
print("\nStep 2: Identifying task-relevant features...")
yes_mean = yes_acts.mean(axis=0)
no_mean = no_acts.mean(axis=0)
diff = yes_mean - no_mean

top_yes_idx = np.argsort(diff)[::-1][:10]
top_no_idx = np.argsort(diff)[:10]

print("Top 10 valid-translation (yes)-responsible features (idx | yes_mean | no_mean | diff):")
for rank, fi in enumerate(top_yes_idx):
    print(f"  #{rank+1}: F{fi} | {yes_mean[fi]:.4f} | {no_mean[fi]:.4f} | {diff[fi]:+.4f}")
print("Top 10 invalid-translation (no)-responsible features:")
for rank, fi in enumerate(top_no_idx):
    print(f"  #{rank+1}: F{fi} | {yes_mean[fi]:.4f} | {no_mean[fi]:.4f} | {diff[fi]:+.4f}")


# ── Plot ─────────────────────────────────────────────────────
print("\nStep 3: Extracting activations for plot pairs...")
yes_plot_acts = np.array([get_sae_activation(en, fr) for en, fr, _ in yes_plot])
no_plot_acts = np.array([get_sae_activation(en, fr) for en, fr, _ in no_plot])


def plot_avg_features(feat_indices, feature_type, yes_mat, no_mat, model_name):
    yes_avg = yes_mat[:, feat_indices].mean(axis=1)
    no_avg = no_mat[:, feat_indices].mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ymax = max(yes_avg.max(), no_avg.max()) * 1.15 or 1
    x_y, x_n = np.arange(len(yes_avg)), np.arange(len(no_avg))

    ax1.bar(x_y, yes_avg, color='#2ecc71', edgecolor='black', linewidth=0.5, width=0.8)
    ax1.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax1.set_title(f'{model_name}: Avg Top-5 {feature_type} features on Predicted-YES pairs', fontsize=13)
    ax1.set_ylim(0, ymax)
    ax1.axhline(y=yes_avg.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean={yes_avg.mean():.4f}')
    ax1.set_xticks(x_y)
    ax1.set_xticklabels([f'Y{i+1}' for i in range(len(yes_avg))], fontsize=8, rotation=45)
    ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x_n, no_avg, color='#e74c3c', edgecolor='black', linewidth=0.5, width=0.8)
    ax2.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax2.set_title(f'{model_name}: Avg Top-5 {feature_type} features on Predicted-NO pairs', fontsize=13)
    ax2.set_ylim(0, ymax)
    ax2.axhline(y=no_avg.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean={no_avg.mean():.4f}')
    ax2.set_xticks(x_n)
    ax2.set_xticklabels([f'N{i+1}' for i in range(len(no_avg))], fontsize=8, rotation=45)
    ax2.set_xlabel('EN-FR pairs', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig, yes_avg, no_avg


print("\nStep 4: Generating plots...")
fig, y_avg_p, n_avg_p = plot_avg_features(
    top_yes_idx[:5], "Valid-translation",
    yes_plot_acts, no_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_qwen3_8b_mt_yes_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

fig, y_avg_n, n_avg_n = plot_avg_features(
    top_no_idx[:5], "Invalid-translation",
    yes_plot_acts, no_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_qwen3_8b_mt_no_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── CSVs ─────────────────────────────────────────────────────
import csv
for tag, feat_ids, colname in [
    ("yes", top_yes_idx[:5], "avg_top5_valid_translation_features"),
    ("no",  top_no_idx[:5],  "avg_top5_invalid_translation_features"),
]:
    vals_y = yes_plot_acts[:, feat_ids].mean(axis=1)
    vals_n = no_plot_acts[:, feat_ids].mean(axis=1)
    csv_path = os.path.join(OUT_DIR, f"firing_values_qwen3_8b_mt_{tag}_avg5.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "pred_set", "english", "french", "ground_truth_label", colname])
        for i, (en, fr, gt) in enumerate(yes_plot):
            w.writerow([f"Y{i+1}", "predicted_yes", en, fr, gt, f"{vals_y[i]:.6f}"])
        for i, (en, fr, gt) in enumerate(no_plot):
            w.writerow([f"N{i+1}", "predicted_no", en, fr, gt, f"{vals_n[i]:.6f}"])
    print(f"CSV: {csv_path}")

for tag, sel, plt_lst in [("yes", yes_select, yes_plot), ("no", no_select, no_plot)]:
    csv_path = os.path.join(OUT_DIR, f"filtered_pairs_qwen3_8b_mt_{tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["idx", "usage", "english", "french", "ground_truth_label"])
        for i, (en, fr, gt) in enumerate(sel):
            w.writerow([i+1, "select", en, fr, gt])
        for i, (en, fr, gt) in enumerate(plt_lst):
            w.writerow([len(sel)+i+1, "plot", en, fr, gt])
    print(f"CSV: {csv_path}")

np.savez(
    os.path.join(OUT_DIR, f"top_features_qwen3_8b_mt_L{LAYER}.npz"),
    top_yes_idx=top_yes_idx.astype(np.int32),
    top_no_idx=top_no_idx.astype(np.int32),
    yes_mean=yes_mean.astype(np.float32),
    no_mean=no_mean.astype(np.float32),
    diff=diff.astype(np.float32),
    layer=LAYER, d_sae=d_sae,
)

print(f"\n{'='*70}")
print(f"VALUES: Avg Top-5 Valid-translation features ({MODEL_NAME})")
print(f"Features: {list(int(x) for x in top_yes_idx[:5])}")
print(f"Predicted-YES mean={y_avg_p.mean():.4f}, Predicted-NO mean={n_avg_p.mean():.4f}")
print(f"\nVALUES: Avg Top-5 Invalid-translation features ({MODEL_NAME})")
print(f"Features: {list(int(x) for x in top_no_idx[:5])}")
print(f"Predicted-YES mean={y_avg_n.mean():.4f}, Predicted-NO mean={n_avg_n.mean():.4f}")
print(f"{'='*70}\nDone.")
