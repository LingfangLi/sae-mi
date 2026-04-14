# -*- coding: utf-8 -*-
"""Feature firing pattern visualization for DeepSeek on SST-2

Shows how task-relevant SAE features fire across positive vs negative sentences.
- Identifies top positive-responsible and negative-responsible features
- Plots bar charts: feature activation intensity across 30 pos / 30 neg sentences

Output: firing_pattern_positive.png, firing_pattern_negative.png
"""

import os, random
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load SST-2 ────────────────────────────────────────────────
from datasets import load_dataset

print("Loading SST-2...")
ds = load_dataset("glue", "sst2")
val = ds["validation"]
train = ds["train"]

# Collect sentences with labels
all_sents = [(ex["sentence"], ex["label"]) for ex in val]
for i in range(min(500, len(train))):
    all_sents.append((train[i]["sentence"], train[i]["label"]))

pos_sents = [(s, l) for s, l in all_sents if l == 1]
neg_sents = [(s, l) for s, l in all_sents if l == 0]
random.seed(42)
random.shuffle(pos_sents)
random.shuffle(neg_sents)

# Both pos and neg will be filtered by zero-shot prediction
N_SELECT = 200
N_PLOT = 30

pos_candidates = pos_sents  # all available, will be filtered by zero-shot
neg_candidates = neg_sents  # all available, will be filtered by zero-shot

print(f"Candidates: {len(pos_candidates)} pos, {len(neg_candidates)} neg")
print("(Both will be filtered by zero-shot prediction after model loads)")

# ── Load Model + SAE ──────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
)
model.eval()

LAYER = 14
sae = Sae.load_from_hub("EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k",
                         hookpoint=f"layers.{LAYER}.mlp")
sae.to(device).eval()
num_latents = sae.num_latents
print(f"SAE: layer={LAYER}, num_latents={num_latents}, TopK={sae.cfg.k}")

# ── Zero-shot sentiment prediction (same as project's base model test) ──
def predict_sentiment_zeroshot(sentence):
    """Use DeepSeek zero-shot to predict positive/negative sentiment.
    Uses next-token logits (same method as deepseek_r1_1_5b_pretrained.py).
    Returns 1 (positive) or 0 (negative).
    """
    prompt = f"""Classify the sentiment as positive or negative.

Sentence: {sentence}
Sentiment:"""
    enc = tokenizer(prompt, return_tensors="pt", padding=False,
                    truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    next_token_logits = out.logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    generated_text = tokenizer.decode([next_token_id]).strip().lower()

    if "positive" in generated_text or generated_text.startswith("pos"):
        return 1
    elif "negative" in generated_text or generated_text.startswith("neg"):
        return 0
    else:
        # Fallback: compare logits for " positive" vs " negative"
        pos_id = tokenizer.encode(" positive")[-1]
        neg_id = tokenizer.encode(" negative")[-1]
        if next_token_logits[pos_id] > next_token_logits[neg_id]:
            return 1
        else:
            return 0

# ── Filter both pos and neg sentences by zero-shot prediction ─────────
print("\nStep 0: Filtering sentences by zero-shot prediction...")
print("  (Using same prompt as deepseek_r1_1_5b_pretrained.py)")

# Filter positive sentences
print("\n  --- Positive sentences ---")
pos_select = []
pos_plot = []
pos_plot_done = False

for i, (s, l) in enumerate(pos_candidates):
    pred = predict_sentiment_zeroshot(s)
    if pred == 1:  # model predicts positive
        if len(pos_select) < N_SELECT:
            pos_select.append((s, l))
        elif not pos_plot_done and len(pos_plot) < N_PLOT:
            pos_plot.append((s, l))
            if len(pos_plot) >= N_PLOT:
                pos_plot_done = True
    if (i + 1) % 20 == 0:
        print(f"  Checked {i+1}/{len(pos_candidates)}: {len(pos_select)} select, {len(pos_plot)} plot accepted")
    if len(pos_select) >= N_SELECT and pos_plot_done:
        break

if len(pos_plot) < N_PLOT:
    extra_needed = N_PLOT - len(pos_plot)
    pos_plot.extend(pos_select[:extra_needed])
    print(f"  Warning: borrowing {extra_needed} from selection set for pos plot")

# Filter negative sentences
print("\n  --- Negative sentences ---")
neg_select = []
neg_plot = []
neg_plot_done = False

for i, (s, l) in enumerate(neg_candidates):
    pred = predict_sentiment_zeroshot(s)
    if pred == 0:  # model predicts negative
        if len(neg_select) < N_SELECT:
            neg_select.append((s, l))
        elif not neg_plot_done and len(neg_plot) < N_PLOT:
            neg_plot.append((s, l))
            if len(neg_plot) >= N_PLOT:
                neg_plot_done = True
    if (i + 1) % 20 == 0:
        print(f"  Checked {i+1}/{len(neg_candidates)}: {len(neg_select)} select, {len(neg_plot)} plot accepted")
    if len(neg_select) >= N_SELECT and neg_plot_done:
        break

if len(neg_plot) < N_PLOT:
    extra_needed = N_PLOT - len(neg_plot)
    neg_plot.extend(neg_select[:extra_needed])
    print(f"  Warning: borrowing {extra_needed} from selection set for neg plot")

print(f"\nAfter zero-shot filtering:")
print(f"  Positive select: {len(pos_select)} (from {len(pos_candidates)} candidates)")
print(f"  Positive plot: {len(pos_plot)}")
print(f"  Negative select: {len(neg_select)} (from {len(neg_candidates)} candidates)")
print(f"  Negative plot: {len(neg_plot)}")

# ── Hook ──────────────────────────────────────────────────────
captured = {}
def hook_fn(module, inp, out):
    captured['out'] = out
hook = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)

# ── Extract activations ──────────────────────────────────────
def get_sae_activation(sentence):
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        model(**enc, use_cache=False)
    mlp_out = captured['out']
    flat = mlp_out.view(-1, mlp_out.shape[-1])
    enc_output = sae.encode(flat)
    dense = torch.zeros(flat.shape[0], num_latents, device=device, dtype=torch.float32)
    dense.scatter_(1, enc_output.top_indices.long(), enc_output.top_acts.float())
    pooled = dense.mean(dim=0).detach().cpu().numpy()
    return pooled

# Step 1: Extract activations for feature selection
print(f"\nStep 1: Extracting activations for feature selection "
      f"({len(pos_select)} pos, {len(neg_select)} neg filtered)...")
pos_acts = []
for i, (s, l) in enumerate(pos_select):
    pos_acts.append(get_sae_activation(s))
    if (i + 1) % 50 == 0:
        print(f"  Pos: {i+1}/{len(pos_select)}")
pos_acts = np.array(pos_acts)  # (N_SELECT, num_latents)

neg_acts = []
for i, (s, l) in enumerate(neg_select):
    neg_acts.append(get_sae_activation(s))
    if (i + 1) % 50 == 0:
        print(f"  Neg: {i+1}/{len(neg_select)}")
neg_acts = np.array(neg_acts)  # (N_SELECT, num_latents)

# Step 2: Find task-relevant features
print("\nStep 2: Identifying task-relevant features...")
pos_mean = pos_acts.mean(axis=0)
neg_mean = neg_acts.mean(axis=0)
diff = pos_mean - neg_mean  # positive = fires more on positive sentiment

# Top positive-responsible features (fire more on positive)
top_pos_idx = np.argsort(diff)[::-1][:10]
# Top negative-responsible features (fire more on negative)
top_neg_idx = np.argsort(diff)[:10]

print(f"\nTop 10 positive-responsible features:")
for rank, fi in enumerate(top_pos_idx):
    print(f"  #{rank+1}: Feature {fi} | pos_mean={pos_mean[fi]:.4f}, neg_mean={neg_mean[fi]:.4f}, diff={diff[fi]:.4f}")

print(f"\nTop 10 negative-responsible features:")
for rank, fi in enumerate(top_neg_idx):
    print(f"  #{rank+1}: Feature {fi} | pos_mean={pos_mean[fi]:.4f}, neg_mean={neg_mean[fi]:.4f}, diff={diff[fi]:+.4f}")

# Step 3: Extract activations for plot sentences
print("\nStep 3: Extracting activations for plot sentences...")
pos_plot_acts = []
for i, (s, l) in enumerate(pos_plot):
    pos_plot_acts.append(get_sae_activation(s))
pos_plot_acts = np.array(pos_plot_acts)

neg_plot_acts = []
for i, (s, l) in enumerate(neg_plot):
    neg_plot_acts.append(get_sae_activation(s))
neg_plot_acts = np.array(neg_plot_acts)

hook.remove()

# ── Plot functions ────────────────────────────────────────────
def plot_single_feature(feat_idx, feat_rank, sentiment_type, pos_vals, neg_vals,
                        pos_sentences, neg_sentences):
    """Plot a single feature's firing pattern: pos sentences vs neg sentences"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    x_pos = np.arange(len(pos_vals))
    x_neg = np.arange(len(neg_vals))

    # Consistent y-axis
    ymax = max(pos_vals.max(), neg_vals.max()) * 1.15
    if ymax == 0:
        ymax = 1

    # Plot on positive sentences
    colors_pos = ['#2ecc71' if v > 0 else '#bdc3c7' for v in pos_vals]
    ax1.bar(x_pos, pos_vals, color=colors_pos, edgecolor='black', linewidth=0.5, width=0.8)
    ax1.set_ylabel('Activation Intensity', fontsize=12)
    ax1.set_title(f'Feature {feat_idx} ({sentiment_type}-responsible, rank #{feat_rank}) '
                  f'on Positive Sentences', fontsize=13)
    ax1.set_ylim(0, ymax)
    ax1.axhline(y=pos_vals.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean={pos_vals.mean():.4f}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'P{i+1}' for i in range(len(pos_vals))], fontsize=8, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Plot on negative sentences
    colors_neg = ['#e74c3c' if v > 0 else '#bdc3c7' for v in neg_vals]
    ax2.bar(x_neg, neg_vals, color=colors_neg, edgecolor='black', linewidth=0.5, width=0.8)
    ax2.set_ylabel('Activation Intensity', fontsize=12)
    ax2.set_title(f'Feature {feat_idx} ({sentiment_type}-responsible, rank #{feat_rank}) '
                  f'on Negative Sentences', fontsize=13)
    ax2.set_ylim(0, ymax)
    ax2.axhline(y=neg_vals.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean={neg_vals.mean():.4f}')
    ax2.set_xticks(x_neg)
    ax2.set_xticklabels([f'N{i+1}' for i in range(len(neg_vals))], fontsize=8, rotation=45)
    ax2.set_xlabel('Sentences', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_avg_features(feat_indices, sentiment_type, pos_acts_mat, neg_acts_mat):
    """Plot average activation of multiple features"""
    pos_avg = pos_acts_mat[:, feat_indices].mean(axis=1)  # avg across features per sentence
    neg_avg = neg_acts_mat[:, feat_indices].mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    ymax = max(pos_avg.max(), neg_avg.max()) * 1.15
    if ymax == 0:
        ymax = 1

    x_pos = np.arange(len(pos_avg))
    x_neg = np.arange(len(neg_avg))

    ax1.bar(x_pos, pos_avg, color='#2ecc71', edgecolor='black', linewidth=0.5, width=0.8)
    ax1.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax1.set_title(f'Average of Top-5 {sentiment_type}-responsible features on Positive Sentences',
                  fontsize=13)
    ax1.set_ylim(0, ymax)
    ax1.axhline(y=pos_avg.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean={pos_avg.mean():.4f}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'P{i+1}' for i in range(len(pos_avg))], fontsize=8, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x_neg, neg_avg, color='#e74c3c', edgecolor='black', linewidth=0.5, width=0.8)
    ax2.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax2.set_title(f'Average of Top-5 {sentiment_type}-responsible features on Negative Sentences',
                  fontsize=13)
    ax2.set_ylim(0, ymax)
    ax2.axhline(y=neg_avg.mean(), color='red', linestyle='--', linewidth=1.5,
                label=f'Mean={neg_avg.mean():.4f}')
    ax2.set_xticks(x_neg)
    ax2.set_xticklabels([f'N{i+1}' for i in range(len(neg_avg))], fontsize=8, rotation=45)
    ax2.set_xlabel('Sentences', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


# ── Generate plots ────────────────────────────────────────────
print("\nStep 4: Generating plots...")

# A. Top-1 positive feature: single feature firing pattern
fi = top_pos_idx[0]
fig = plot_single_feature(fi, 1, "Positive",
                          pos_plot_acts[:, fi], neg_plot_acts[:, fi],
                          [s for s, l in pos_plot], [s for s, l in neg_plot])
out = os.path.join(SCRIPT_DIR, "firing_pattern_pos_top1.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# B. Top-1 negative feature: single feature firing pattern
fi = top_neg_idx[0]
fig = plot_single_feature(fi, 1, "Negative",
                          pos_plot_acts[:, fi], neg_plot_acts[:, fi],
                          [s for s, l in pos_plot], [s for s, l in neg_plot])
out = os.path.join(SCRIPT_DIR, "firing_pattern_neg_top1.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# C. Average of top-5 positive features
fig = plot_avg_features(top_pos_idx[:5], "Positive", pos_plot_acts, neg_plot_acts)
out = os.path.join(SCRIPT_DIR, "firing_pattern_pos_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# D. Average of top-5 negative features
fig = plot_avg_features(top_neg_idx[:5], "Negative", pos_plot_acts, neg_plot_acts)
out = os.path.join(SCRIPT_DIR, "firing_pattern_neg_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Print example sentences for reference ─────────────────────
print(f"\n{'='*70}")
print("PLOT SENTENCES (for reference)")
print(f"{'='*70}")
print("\nPositive sentences:")
for i, (s, l) in enumerate(pos_plot):
    print(f"  P{i+1}: {s[:80]}{'...' if len(s) > 80 else ''}")
print("\nNegative sentences:")
for i, (s, l) in enumerate(neg_plot):
    print(f"  N{i+1}: {s[:80]}{'...' if len(s) > 80 else ''}")

# ── Save CSV values for supervisor ─────────────────────────────
import csv

# Top-1 positive feature values
fi = top_pos_idx[0]
csv_path = os.path.join(SCRIPT_DIR, "firing_values_pos_top1.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["sentence_id", "sentence_type", "sentence_text", f"feature_{fi}_activation"])
    for i, (s, l) in enumerate(pos_plot):
        w.writerow([f"P{i+1}", "positive", s, f"{pos_plot_acts[i, fi]:.6f}"])
    for i, (s, l) in enumerate(neg_plot):
        w.writerow([f"N{i+1}", "negative", s, f"{neg_plot_acts[i, fi]:.6f}"])
print(f"CSV: {csv_path}")

# Top-1 negative feature values
fi = top_neg_idx[0]
csv_path = os.path.join(SCRIPT_DIR, "firing_values_neg_top1.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["sentence_id", "sentence_type", "sentence_text", f"feature_{fi}_activation"])
    for i, (s, l) in enumerate(pos_plot):
        w.writerow([f"P{i+1}", "positive", s, f"{pos_plot_acts[i, fi]:.6f}"])
    for i, (s, l) in enumerate(neg_plot):
        w.writerow([f"N{i+1}", "negative", s, f"{neg_plot_acts[i, fi]:.6f}"])
print(f"CSV: {csv_path}")

# Avg of top-5 positive features
csv_path = os.path.join(SCRIPT_DIR, "firing_values_pos_avg5.csv")
pos_avg = pos_plot_acts[:, top_pos_idx[:5]].mean(axis=1)
neg_avg = neg_plot_acts[:, top_pos_idx[:5]].mean(axis=1)
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["sentence_id", "sentence_type", "sentence_text", "avg_top5_pos_features"])
    for i, (s, l) in enumerate(pos_plot):
        w.writerow([f"P{i+1}", "positive", s, f"{pos_avg[i]:.6f}"])
    for i, (s, l) in enumerate(neg_plot):
        w.writerow([f"N{i+1}", "negative", s, f"{neg_avg[i]:.6f}"])
print(f"CSV: {csv_path}")

# Avg of top-5 negative features
csv_path = os.path.join(SCRIPT_DIR, "firing_values_neg_avg5.csv")
pos_avg_n = pos_plot_acts[:, top_neg_idx[:5]].mean(axis=1)
neg_avg_n = neg_plot_acts[:, top_neg_idx[:5]].mean(axis=1)
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["sentence_id", "sentence_type", "sentence_text", "avg_top5_neg_features"])
    for i, (s, l) in enumerate(pos_plot):
        w.writerow([f"P{i+1}", "positive", s, f"{pos_avg_n[i]:.6f}"])
    for i, (s, l) in enumerate(neg_plot):
        w.writerow([f"N{i+1}", "negative", s, f"{neg_avg_n[i]:.6f}"])
print(f"CSV: {csv_path}")

# Print avg positive feature values
print(f"\n{'='*70}")
print("VALUES: Avg Top-5 Positive Features")
print(f"Features used: {list(top_pos_idx[:5])}")
print(f"{'='*70}")
print(f"\nOn Positive Sentences (mean={pos_avg.mean():.4f}):")
for i in range(len(pos_avg)):
    print(f"  P{i+1}: {pos_avg[i]:.6f}  | {pos_plot[i][0][:60]}")
print(f"\nOn Negative Sentences (mean={neg_avg.mean():.4f}):")
for i in range(len(neg_avg)):
    print(f"  N{i+1}: {neg_avg[i]:.6f}  | {neg_plot[i][0][:60]}")

# Print avg negative feature values
print(f"\n{'='*70}")
print("VALUES: Avg Top-5 Negative Features")
print(f"Features used: {list(top_neg_idx[:5])}")
print(f"{'='*70}")
print(f"\nOn Positive Sentences (mean={pos_avg_n.mean():.4f}):")
for i in range(len(pos_avg_n)):
    print(f"  P{i+1}: {pos_avg_n[i]:.6f}  | {pos_plot[i][0][:60]}")
print(f"\nOn Negative Sentences (mean={neg_avg_n.mean():.4f}):")
for i in range(len(neg_avg_n)):
    print(f"  N{i+1}: {neg_avg_n[i]:.6f}  | {neg_plot[i][0][:60]}")

# ── Save filtered sentence lists ─────────────────────────────
csv_path = os.path.join(SCRIPT_DIR, "filtered_sentences_pos.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["idx", "usage", "sentence", "ground_truth_label"])
    for i, (s, l) in enumerate(pos_select):
        w.writerow([i+1, "select", s, l])
    for i, (s, l) in enumerate(pos_plot):
        w.writerow([len(pos_select)+i+1, "plot", s, l])
print(f"CSV: {csv_path}")

csv_path = os.path.join(SCRIPT_DIR, "filtered_sentences_neg.csv")
with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["idx", "usage", "sentence", "ground_truth_label"])
    for i, (s, l) in enumerate(neg_select):
        w.writerow([i+1, "select", s, l])
    for i, (s, l) in enumerate(neg_plot):
        w.writerow([len(neg_select)+i+1, "plot", s, l])
print(f"CSV: {csv_path}")

print("\n\nDone.")
