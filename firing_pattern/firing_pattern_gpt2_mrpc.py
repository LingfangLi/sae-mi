# -*- coding: utf-8 -*-
"""Firing pattern visualization for GPT-2 Small on MRPC (paraphrase detection)

Question: for semantic-similarity-like SAE features, how do firing patterns
differ between sentence pairs the model predicts as "yes" (paraphrase) vs
"no" (non-paraphrase)?

Pipeline (same shape as firing_pattern_gpt2.py for SST-2):
  1. Split MRPC candidates by the model's zero-shot prediction (NOT by ground truth).
     - yes_select / yes_plot : pairs where zero-shot predicts "yes"
     - no_select  / no_plot  : pairs where zero-shot predicts "no"
  2. Extract SAE activations (attention hook_z, mean-pooled over tokens).
  3. diff = yes_mean - no_mean over features.
     top_yes_idx = top-10 features firing more on predicted-yes   (semantic-similarity-like)
     top_no_idx  = top-10 features firing more on predicted-no
  4. Bar plots: avg of top-5 yes/no features on yes_plot vs no_plot pairs.
  5. Save CSVs of per-pair firing values + filtered candidate lists.

Model: GPT-2 Small (TransformerLens)
SAE:   gpt2-small-hook-z-kk (attention hook_z)
Layer: 2 (probing-best layer under attn.hook_z + mean pool; matches SST-2
       firing_pattern_gpt2.py which uses L7 = SST-2 probing-best under the
       same config)
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
OUT_DIR   = os.path.join(SCRIPT_DIR, "gpt2_mrpc")
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

# Combine val + train as candidate pool. MRPC zero-shot tends to over-predict
# "yes" on GPT-2, so we need a generous candidate pool for the "no" bucket.
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
from transformer_lens import HookedTransformer
from sae_lens import SAE

print("Loading GPT-2 Small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device,
                                           cache_dir=CACHE_DIR)
model.eval()

LAYER = 2
hook_name = f"blocks.{LAYER}.attn.hook_z"
release = "gpt2-small-hook-z-kk"
sae_id = f"blocks.{LAYER}.hook_z"

print(f"Loading SAE: {release}, {sae_id}")
try:
    sae = SAE.from_pretrained(release, sae_id)
except Exception:
    sae = SAE.from_pretrained(release, sae_id)[0]
sae.to(device).eval()
d_sae = sae.cfg.d_sae
print(f"SAE: layer={LAYER}, d_sae={d_sae}")

# ── Combine sentence pair (same as gpt2_sae_mrpc.py) ─────────
def combine_pair(s1, s2):
    return f"{s1} <|endoftext|> {s2}"

# ── Zero-shot paraphrase prediction ──────────────────────────
YES_ID = model.to_single_token(" yes")
NO_ID  = model.to_single_token(" no")

def predict_paraphrase_zeroshot(s1, s2):
    """Returns 1 if model predicts paraphrase, 0 otherwise. Same prompt as
    gpt2_sae_mrpc.py."""
    prompt = f"""Are these two sentences paraphrases?

Sentence 1: {s1}
Sentence 2: {s2}
Answer (yes/no):"""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        logits = model(tokens)
    next_token_logits = logits[0, -1, :]
    return 1 if next_token_logits[YES_ID] > next_token_logits[NO_ID] else 0

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

# If we didn't reach N_PLOT, borrow from select set (warns loudly)
for name, sel, plt_lst in [("yes", yes_select, yes_plot),
                            ("no",  no_select,  no_plot)]:
    if len(plt_lst) < N_PLOT:
        extra = N_PLOT - len(plt_lst)
        plt_lst.extend(sel[:extra])
        print(f"  Warning: borrowing {extra} from {name}_select for {name}_plot")

print(f"\nAfter zero-shot filtering:")
print(f"  Predicted-YES  select: {len(yes_select)}, plot: {len(yes_plot)}")
print(f"  Predicted-NO   select: {len(no_select)},  plot: {len(no_plot)}")

# ── Extract SAE activations ──────────────────────────────────
def get_sae_activation(s1, s2):
    text = combine_pair(s1, s2)
    tokens = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
    hook_acts = cache[hook_name]
    sae_feature_acts = sae.encode(hook_acts)
    pooled = sae_feature_acts.mean(dim=1)[0].detach().cpu().numpy()
    return pooled

print(f"\nStep 1: Extracting activations "
      f"({len(yes_select)} yes, {len(no_select)} no)...")
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

# ── Feature selection (task-relevant / semantic-similarity) ──
print("\nStep 2: Identifying task-relevant features...")
yes_mean = yes_acts.mean(axis=0)
no_mean  = no_acts.mean(axis=0)
diff = yes_mean - no_mean  # >0 fires more on predicted-paraphrase

top_yes_idx = np.argsort(diff)[::-1][:10]   # "semantic-similarity"-like
top_no_idx  = np.argsort(diff)[:10]          # features firing more on non-paraphrase

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

# ── Extract activations for plot pairs ────────────────────────
print("\nStep 3: Extracting activations for plot pairs...")
yes_plot_acts = np.array([get_sae_activation(s1, s2) for s1, s2, _ in yes_plot])
no_plot_acts  = np.array([get_sae_activation(s1, s2) for s1, s2, _ in no_plot])

# ── Plot ──────────────────────────────────────────────────────
def plot_avg_features(feat_indices, feature_type, yes_mat, no_mat, model_name):
    """feature_type: 'Paraphrase (semantic-similarity)' or 'Non-paraphrase'"""
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
MODEL_NAME = f"GPT-2 Small (MRPC, L{LAYER})"

fig, y_avg_p, n_avg_p = plot_avg_features(
    top_yes_idx[:5], "Paraphrase (semantic-sim)",
    yes_plot_acts, no_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_gpt2_mrpc_yes_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

fig, y_avg_n, n_avg_n = plot_avg_features(
    top_no_idx[:5], "Non-paraphrase",
    yes_plot_acts, no_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_gpt2_mrpc_no_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Save CSVs ─────────────────────────────────────────────────
import csv

for tag, feat_ids, colname in [
    ("yes", top_yes_idx[:5], "avg_top5_paraphrase_features"),
    ("no",  top_no_idx[:5],  "avg_top5_nonparaphrase_features")
]:
    vals_y = yes_plot_acts[:, feat_ids].mean(axis=1)
    vals_n = no_plot_acts[:,  feat_ids].mean(axis=1)
    csv_path = os.path.join(OUT_DIR,
                            f"firing_values_gpt2_mrpc_{tag}_avg5.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "pred_set", "sentence1", "sentence2",
                    "ground_truth_label", colname])
        for i, (s1, s2, gt) in enumerate(yes_plot):
            w.writerow([f"Y{i+1}", "predicted_yes", s1, s2, gt,
                        f"{vals_y[i]:.6f}"])
        for i, (s1, s2, gt) in enumerate(no_plot):
            w.writerow([f"N{i+1}", "predicted_no", s1, s2, gt,
                        f"{vals_n[i]:.6f}"])
    print(f"CSV: {csv_path}")

for tag, sel, plt_lst in [("yes", yes_select, yes_plot),
                           ("no",  no_select,  no_plot)]:
    csv_path = os.path.join(OUT_DIR,
                            f"filtered_pairs_gpt2_mrpc_{tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["idx", "usage", "sentence1", "sentence2",
                    "ground_truth_label"])
        for i, (s1, s2, gt) in enumerate(sel):
            w.writerow([i+1, "select", s1, s2, gt])
        for i, (s1, s2, gt) in enumerate(plt_lst):
            w.writerow([len(sel)+i+1, "plot", s1, s2, gt])
    print(f"CSV: {csv_path}")

# ── Summary ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"VALUES: Avg Top-5 Paraphrase (semantic-sim) Features "
      f"(GPT-2, MRPC, L{LAYER})")
print(f"Features: {list(int(x) for x in top_yes_idx[:5])}")
print(f"Predicted-YES mean={y_avg_p.mean():.4f}, "
      f"Predicted-NO mean={n_avg_p.mean():.4f}")
print(f"\nVALUES: Avg Top-5 Non-paraphrase Features (GPT-2, MRPC, L{LAYER})")
print(f"Features: {list(int(x) for x in top_no_idx[:5])}")
print(f"Predicted-YES mean={y_avg_n.mean():.4f}, "
      f"Predicted-NO mean={n_avg_n.mean():.4f}")
print(f"{'='*70}")

print("\n\nDone.")
