# -*- coding: utf-8 -*-
"""Firing pattern visualization for Qwen3-0.6B on MRPC (paraphrase detection).

Two sample sets split by the model's zero-shot prediction (not GT).
Identify top-10 features that fire more on predicted-paraphrase vs predicted-
non-paraphrase; plot their average firing pattern on 30 held-out pairs per set.

Model: Qwen3-0.6B-Base (HF Transformers)
SAE:   mwhanna-qwen3-0.6b-transcoders-lowl0 (Transcoder)
Layer: 14  (mean-pool probing-best on MRPC, from
       saes-mi/Mechanistic-Interpretability/Pre-Trained Models/MRPC/Qwen3 0.6B/
       final_summary.txt: L14 = 74.75%)
Pool:  Mean
"""

import os, random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
OUT_DIR   = os.path.join(SCRIPT_DIR, "qwen3_mrpc")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load MRPC ─────────────────────────────────────────────────
from datasets import load_dataset

print("Loading MRPC...")
ds = load_dataset("glue", "mrpc")
val = ds["validation"]
train = ds["train"]

all_pairs = [(ex["sentence1"], ex["sentence2"], ex["label"]) for ex in val]
for i in range(min(2000, len(train))):
    ex = train[i]
    all_pairs.append((ex["sentence1"], ex["sentence2"], ex["label"]))

random.seed(42)
random.shuffle(all_pairs)

N_SELECT = 200
N_PLOT = 30
print(f"Candidate pool: {len(all_pairs)} pairs")

# ── Load Model + SAE ──────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

base_model_id = "Qwen/Qwen3-0.6B-Base"
print(f"Loading {base_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
)
model.eval()

LAYER = 14
release = "mwhanna-qwen3-0.6b-transcoders-lowl0"
sae_id = f"layer_{LAYER}"

print(f"Loading SAE: {release}, {sae_id}")
try:
    sae = SAE.from_pretrained(release, sae_id)
except Exception:
    sae = SAE.from_pretrained(release, sae_id)[0]
sae.to(device).eval()
d_sae = sae.cfg.d_sae
print(f"SAE: layer={LAYER}, d_sae={d_sae}")

# ── Pair combine ─────────────────────────────────────────────
EOS = "<|endoftext|>"
def combine_pair(s1, s2):
    return f"{s1} {EOS} {s2}"

# ── Hidden-states helper (HF) ────────────────────────────────
def get_layer_rep(text, layer_idx):
    enc = tokenizer(text, return_tensors="pt", padding=False,
                    truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx + 1]  # (1, seq, d_model)

# ── Zero-shot paraphrase prediction ──────────────────────────
def predict_paraphrase_zeroshot(s1, s2):
    prompt = f"""Are these two sentences paraphrases?

Sentence 1: {s1}
Sentence 2: {s2}
Answer (yes/no):"""
    enc = tokenizer(prompt, return_tensors="pt", padding=False,
                    truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    next_token_logits = out.logits[0, -1, :]
    yes_id = tokenizer.encode(" yes")[-1]
    no_id  = tokenizer.encode(" no")[-1]
    return 1 if next_token_logits[yes_id] > next_token_logits[no_id] else 0

# ── Filter pairs by zero-shot prediction ──────────────────────
print("\nStep 0: Filtering pairs by zero-shot prediction (by model, not GT)...")
yes_select, yes_plot = [], []
no_select,  no_plot  = [], []
yes_plot_done = False
no_plot_done  = False

for i, (s1, s2, gt) in enumerate(all_pairs):
    if yes_plot_done and no_plot_done \
       and len(yes_select) >= N_SELECT and len(no_select) >= N_SELECT:
        break
    pred = predict_paraphrase_zeroshot(s1, s2)
    if pred == 1:
        if len(yes_select) < N_SELECT:
            yes_select.append((s1, s2, gt))
        elif not yes_plot_done and len(yes_plot) < N_PLOT:
            yes_plot.append((s1, s2, gt))
            if len(yes_plot) >= N_PLOT:
                yes_plot_done = True
    else:
        if len(no_select) < N_SELECT:
            no_select.append((s1, s2, gt))
        elif not no_plot_done and len(no_plot) < N_PLOT:
            no_plot.append((s1, s2, gt))
            if len(no_plot) >= N_PLOT:
                no_plot_done = True
    if (i + 1) % 100 == 0:
        print(f"  Checked {i+1}/{len(all_pairs)}: "
              f"yes_select={len(yes_select)}, yes_plot={len(yes_plot)}, "
              f"no_select={len(no_select)}, no_plot={len(no_plot)}")

for name, sel, plt_lst in [("yes", yes_select, yes_plot),
                            ("no",  no_select,  no_plot)]:
    if len(plt_lst) < N_PLOT:
        extra = N_PLOT - len(plt_lst)
        plt_lst.extend(sel[:extra])
        print(f"  Warning: borrowing {extra} from {name}_select for {name}_plot")

print(f"\nAfter filtering:")
print(f"  Predicted-YES select={len(yes_select)}, plot={len(yes_plot)}")
print(f"  Predicted-NO  select={len(no_select)},  plot={len(no_plot)}")

# ── Extract SAE activations ──────────────────────────────────
def get_sae_activation(s1, s2):
    text = combine_pair(s1, s2)
    layer_h = get_layer_rep(text, LAYER)
    sae_feats = sae.encode(layer_h)
    pooled = sae_feats.mean(dim=1)[0].to(torch.float32).detach().cpu().numpy()
    return pooled

print(f"\nStep 1: Extracting activations ({len(yes_select)} yes, {len(no_select)} no)...")
yes_acts = []
for i, (s1, s2, gt) in enumerate(yes_select):
    yes_acts.append(get_sae_activation(s1, s2))
    if (i + 1) % 50 == 0:
        print(f"  Yes: {i+1}/{len(yes_select)}")
yes_acts = np.array(yes_acts)

no_acts = []
for i, (s1, s2, gt) in enumerate(no_select):
    no_acts.append(get_sae_activation(s1, s2))
    if (i + 1) % 50 == 0:
        print(f"  No:  {i+1}/{len(no_select)}")
no_acts = np.array(no_acts)

# ── Feature selection ────────────────────────────────────────
print("\nStep 2: Identifying task-relevant features...")
yes_mean = yes_acts.mean(axis=0)
no_mean  = no_acts.mean(axis=0)
diff = yes_mean - no_mean

top_yes_idx = np.argsort(diff)[::-1][:10]
top_no_idx  = np.argsort(diff)[:10]

print(f"\nTop 10 paraphrase-responsible features (semantic-similarity):")
for rank, fi in enumerate(top_yes_idx):
    print(f"  #{rank+1}: Feature {fi} | "
          f"yes_mean={yes_mean[fi]:.4f}, no_mean={no_mean[fi]:.4f}, "
          f"diff={diff[fi]:+.4f}")

print(f"\nTop 10 non-paraphrase-responsible features:")
for rank, fi in enumerate(top_no_idx):
    print(f"  #{rank+1}: Feature {fi} | "
          f"yes_mean={yes_mean[fi]:.4f}, no_mean={no_mean[fi]:.4f}, "
          f"diff={diff[fi]:+.4f}")

# ── Plot pair activations ────────────────────────────────────
print("\nStep 3: Extracting activations for plot pairs...")
yes_plot_acts = np.array([get_sae_activation(s1, s2) for s1, s2, _ in yes_plot])
no_plot_acts  = np.array([get_sae_activation(s1, s2) for s1, s2, _ in no_plot])

def plot_avg_features(feat_indices, feature_type, yes_mat, no_mat, model_name):
    yes_avg = yes_mat[:, feat_indices].mean(axis=1)
    no_avg  = no_mat[:,  feat_indices].mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ymax = max(yes_avg.max(), no_avg.max()) * 1.15
    if ymax == 0:
        ymax = 1
    x_y = np.arange(len(yes_avg))
    x_n = np.arange(len(no_avg))
    ax1.bar(x_y, yes_avg, color='#2ecc71', edgecolor='black',
            linewidth=0.5, width=0.8)
    ax1.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax1.set_title(f'{model_name}: Avg Top-5 {feature_type} features '
                  f'on Predicted-YES pairs', fontsize=13)
    ax1.set_ylim(0, ymax)
    ax1.axhline(y=yes_avg.mean(), color='red', linestyle='--',
                linewidth=1.5, label=f'Mean={yes_avg.mean():.4f}')
    ax1.set_xticks(x_y)
    ax1.set_xticklabels([f'Y{i+1}' for i in range(len(yes_avg))],
                        fontsize=8, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax2.bar(x_n, no_avg, color='#e74c3c', edgecolor='black',
            linewidth=0.5, width=0.8)
    ax2.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax2.set_title(f'{model_name}: Avg Top-5 {feature_type} features '
                  f'on Predicted-NO pairs', fontsize=13)
    ax2.set_ylim(0, ymax)
    ax2.axhline(y=no_avg.mean(), color='red', linestyle='--',
                linewidth=1.5, label=f'Mean={no_avg.mean():.4f}')
    ax2.set_xticks(x_n)
    ax2.set_xticklabels([f'N{i+1}' for i in range(len(no_avg))],
                        fontsize=8, rotation=45)
    ax2.set_xlabel('Sentence pairs', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig, yes_avg, no_avg

print("\nStep 4: Generating plots...")
MODEL_NAME = f"Qwen3-0.6B (MRPC, L{LAYER})"

fig, y_avg_p, n_avg_p = plot_avg_features(
    top_yes_idx[:5], "Paraphrase (semantic-sim)",
    yes_plot_acts, no_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_qwen3_mrpc_yes_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

fig, y_avg_n, n_avg_n = plot_avg_features(
    top_no_idx[:5], "Non-paraphrase",
    yes_plot_acts, no_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_qwen3_mrpc_no_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── CSVs ─────────────────────────────────────────────────────
import csv
for tag, feat_ids, colname in [
    ("yes", top_yes_idx[:5], "avg_top5_paraphrase_features"),
    ("no",  top_no_idx[:5],  "avg_top5_nonparaphrase_features")
]:
    vals_y = yes_plot_acts[:, feat_ids].mean(axis=1)
    vals_n = no_plot_acts[:,  feat_ids].mean(axis=1)
    csv_path = os.path.join(OUT_DIR, f"firing_values_qwen3_mrpc_{tag}_avg5.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "pred_set", "sentence1", "sentence2",
                    "ground_truth_label", colname])
        for i, (s1, s2, gt) in enumerate(yes_plot):
            w.writerow([f"Y{i+1}", "predicted_yes", s1, s2, gt, f"{vals_y[i]:.6f}"])
        for i, (s1, s2, gt) in enumerate(no_plot):
            w.writerow([f"N{i+1}", "predicted_no",  s1, s2, gt, f"{vals_n[i]:.6f}"])
    print(f"CSV: {csv_path}")

for tag, sel, plt_lst in [("yes", yes_select, yes_plot),
                           ("no",  no_select,  no_plot)]:
    csv_path = os.path.join(OUT_DIR, f"filtered_pairs_qwen3_mrpc_{tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["idx", "usage", "sentence1", "sentence2", "ground_truth_label"])
        for i, (s1, s2, gt) in enumerate(sel):
            w.writerow([i+1, "select", s1, s2, gt])
        for i, (s1, s2, gt) in enumerate(plt_lst):
            w.writerow([len(sel)+i+1, "plot", s1, s2, gt])
    print(f"CSV: {csv_path}")

print(f"\n{'='*70}")
print(f"VALUES: Avg Top-5 Paraphrase (semantic-sim) (Qwen3-0.6B, MRPC, L{LAYER})")
print(f"Features: {list(int(x) for x in top_yes_idx[:5])}")
print(f"Predicted-YES mean={y_avg_p.mean():.4f}, Predicted-NO mean={n_avg_p.mean():.4f}")
print(f"\nVALUES: Avg Top-5 Non-paraphrase (Qwen3-0.6B, MRPC, L{LAYER})")
print(f"Features: {list(int(x) for x in top_no_idx[:5])}")
print(f"Predicted-YES mean={y_avg_n.mean():.4f}, Predicted-NO mean={n_avg_n.mean():.4f}")
print(f"{'='*70}")

print("\n\nDone.")
