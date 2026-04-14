# -*- coding: utf-8 -*-
"""Firing pattern visualization for Gemma-2-2B on SST-2

Model: Gemma-2-2B (TransformerLens)
SAE: gemma-scope-2b-pt-att-canonical (attention hook_z, 16k), mean pooling
Layer: 12
Zero-shot: same prompt as pretrained_Gemma2b_sae_att.py
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

all_sents = [(ex["sentence"], ex["label"]) for ex in val]
for i in range(min(500, len(train))):
    all_sents.append((train[i]["sentence"], train[i]["label"]))

pos_sents = [(s, l) for s, l in all_sents if l == 1]
neg_sents = [(s, l) for s, l in all_sents if l == 0]
random.seed(42)
random.shuffle(pos_sents)
random.shuffle(neg_sents)

N_SELECT = 200
N_PLOT = 30

pos_candidates = pos_sents
neg_candidates = neg_sents

print(f"Candidates: {len(pos_candidates)} pos, {len(neg_candidates)} neg")

# ── Load Model + SAE ──────────────────────────────────────────
from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged in to HF")

print("Loading Gemma-2-2B...")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device,
                                           cache_dir=CACHE_DIR)
model.eval()

LAYER = 12
hook_name = f"blocks.{LAYER}.attn.hook_z"
release = "gemma-scope-2b-pt-att-canonical"
sae_id = f"layer_{LAYER}/width_16k/canonical"

print(f"Loading SAE: {release}, {sae_id}")
try:
    sae = SAE.from_pretrained(release, sae_id)
except:
    sae = SAE.from_pretrained(release, sae_id)[0]
sae.to(device).eval()
d_sae = sae.cfg.d_sae
print(f"SAE: layer={LAYER}, d_sae={d_sae}")

# ── Zero-shot sentiment prediction ───────────────────────────
def predict_sentiment_zeroshot(sentence):
    """Same method as pretrained_Gemma2b_sae_att.py"""
    prompt = f"""Classify the sentiment as positive or negative.

Sentence: {sentence}
Sentiment:"""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        logits = model(tokens)
    next_token_logits = logits[0, -1, :]
    next_token_id = torch.argmax(next_token_logits).item()
    generated_text = model.tokenizer.decode([next_token_id]).strip().lower()

    if "positive" in generated_text or generated_text.startswith("pos"):
        return 1
    elif "negative" in generated_text or generated_text.startswith("neg"):
        return 0
    else:
        pos_token_id = model.to_tokens(" positive", prepend_bos=False)[0, 0]
        neg_token_id = model.to_tokens(" negative", prepend_bos=False)[0, 0]
        if next_token_logits[pos_token_id] > next_token_logits[neg_token_id]:
            return 1
        else:
            return 0

# ── Filter sentences by zero-shot ─────────────────────────────
print("\nStep 0: Filtering sentences by zero-shot prediction...")

print("\n  --- Positive sentences ---")
pos_select = []
pos_plot = []
pos_plot_done = False
for i, (s, l) in enumerate(pos_candidates):
    pred = predict_sentiment_zeroshot(s)
    if pred == 1:
        if len(pos_select) < N_SELECT:
            pos_select.append((s, l))
        elif not pos_plot_done and len(pos_plot) < N_PLOT:
            pos_plot.append((s, l))
            if len(pos_plot) >= N_PLOT:
                pos_plot_done = True
    if (i + 1) % 20 == 0:
        print(f"  Checked {i+1}/{len(pos_candidates)}: {len(pos_select)} select, {len(pos_plot)} plot")
    if len(pos_select) >= N_SELECT and pos_plot_done:
        break
if len(pos_plot) < N_PLOT:
    extra = N_PLOT - len(pos_plot)
    pos_plot.extend(pos_select[:extra])
    print(f"  Warning: borrowing {extra} from selection set for pos plot")

print("\n  --- Negative sentences ---")
neg_select = []
neg_plot = []
neg_plot_done = False
for i, (s, l) in enumerate(neg_candidates):
    pred = predict_sentiment_zeroshot(s)
    if pred == 0:
        if len(neg_select) < N_SELECT:
            neg_select.append((s, l))
        elif not neg_plot_done and len(neg_plot) < N_PLOT:
            neg_plot.append((s, l))
            if len(neg_plot) >= N_PLOT:
                neg_plot_done = True
    if (i + 1) % 20 == 0:
        print(f"  Checked {i+1}/{len(neg_candidates)}: {len(neg_select)} select, {len(neg_plot)} plot")
    if len(neg_select) >= N_SELECT and neg_plot_done:
        break
if len(neg_plot) < N_PLOT:
    extra = N_PLOT - len(neg_plot)
    neg_plot.extend(neg_select[:extra])
    print(f"  Warning: borrowing {extra} from selection set for neg plot")

print(f"\nAfter zero-shot filtering:")
print(f"  Positive select: {len(pos_select)}, plot: {len(pos_plot)}")
print(f"  Negative select: {len(neg_select)}, plot: {len(neg_plot)}")

# ── Extract SAE activations ──────────────────────────────────
def get_sae_activation(sentence):
    tokens = model.to_tokens(sentence, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    hook_acts = cache[hook_name]
    sae_feature_acts = sae.encode(hook_acts)
    pooled = sae_feature_acts.mean(dim=1)[0].detach().cpu().numpy()
    return pooled

print(f"\nStep 1: Extracting activations ({len(pos_select)} pos, {len(neg_select)} neg)...")
pos_acts = []
for i, (s, l) in enumerate(pos_select):
    pos_acts.append(get_sae_activation(s))
    if (i + 1) % 50 == 0:
        print(f"  Pos: {i+1}/{len(pos_select)}")
pos_acts = np.array(pos_acts)

neg_acts = []
for i, (s, l) in enumerate(neg_select):
    neg_acts.append(get_sae_activation(s))
    if (i + 1) % 50 == 0:
        print(f"  Neg: {i+1}/{len(neg_select)}")
neg_acts = np.array(neg_acts)

# ── Feature selection ─────────────────────────────────────────
print("\nStep 2: Identifying task-relevant features...")
pos_mean = pos_acts.mean(axis=0)
neg_mean = neg_acts.mean(axis=0)
diff = pos_mean - neg_mean

top_pos_idx = np.argsort(diff)[::-1][:10]
top_neg_idx = np.argsort(diff)[:10]

print(f"\nTop 10 positive-responsible features:")
for rank, fi in enumerate(top_pos_idx):
    print(f"  #{rank+1}: Feature {fi} | pos_mean={pos_mean[fi]:.4f}, neg_mean={neg_mean[fi]:.4f}, diff={diff[fi]:.4f}")

print(f"\nTop 10 negative-responsible features:")
for rank, fi in enumerate(top_neg_idx):
    print(f"  #{rank+1}: Feature {fi} | pos_mean={pos_mean[fi]:.4f}, neg_mean={neg_mean[fi]:.4f}, diff={diff[fi]:+.4f}")

# ── Plot activations ──────────────────────────────────────────
print("\nStep 3: Extracting activations for plot sentences...")
pos_plot_acts = np.array([get_sae_activation(s) for s, l in pos_plot])
neg_plot_acts = np.array([get_sae_activation(s) for s, l in neg_plot])

def plot_avg_features(feat_indices, sentiment_type, pos_acts_mat, neg_acts_mat, model_name):
    pos_avg = pos_acts_mat[:, feat_indices].mean(axis=1)
    neg_avg = neg_acts_mat[:, feat_indices].mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ymax = max(pos_avg.max(), neg_avg.max()) * 1.15
    if ymax == 0:
        ymax = 1
    x_pos = np.arange(len(pos_avg))
    x_neg = np.arange(len(neg_avg))

    ax1.bar(x_pos, pos_avg, color='#2ecc71', edgecolor='black', linewidth=0.5, width=0.8)
    ax1.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax1.set_title(f'{model_name}: Avg Top-5 {sentiment_type}-responsible features on Positive Sentences', fontsize=13)
    ax1.set_ylim(0, ymax)
    ax1.axhline(y=pos_avg.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={pos_avg.mean():.4f}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'P{i+1}' for i in range(len(pos_avg))], fontsize=8, rotation=45)
    ax1.legend(fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x_neg, neg_avg, color='#e74c3c', edgecolor='black', linewidth=0.5, width=0.8)
    ax2.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax2.set_title(f'{model_name}: Avg Top-5 {sentiment_type}-responsible features on Negative Sentences', fontsize=13)
    ax2.set_ylim(0, ymax)
    ax2.axhline(y=neg_avg.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={neg_avg.mean():.4f}')
    ax2.set_xticks(x_neg)
    ax2.set_xticklabels([f'N{i+1}' for i in range(len(neg_avg))], fontsize=8, rotation=45)
    ax2.set_xlabel('Sentences', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, pos_avg, neg_avg

print("\nStep 4: Generating plots...")
MODEL_NAME = "Gemma-2-2B"

fig, pos_avg, neg_avg = plot_avg_features(top_pos_idx[:5], "Positive", pos_plot_acts, neg_plot_acts, MODEL_NAME)
out = os.path.join(SCRIPT_DIR, "firing_pattern_gemma_pos_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

fig, neg_avg_n, neg_avg_nn = plot_avg_features(top_neg_idx[:5], "Negative", pos_plot_acts, neg_plot_acts, MODEL_NAME)
out = os.path.join(SCRIPT_DIR, "firing_pattern_gemma_neg_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Save CSV ──────────────────────────────────────────────────
import csv

for tag, feat_ids, label in [("pos", top_pos_idx[:5], "avg_top5_pos_features"),
                               ("neg", top_neg_idx[:5], "avg_top5_neg_features")]:
    vals_p = pos_plot_acts[:, feat_ids].mean(axis=1)
    vals_n = neg_plot_acts[:, feat_ids].mean(axis=1)
    csv_path = os.path.join(SCRIPT_DIR, f"firing_values_gemma_{tag}_avg5.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "sentence_type", "sentence_text", label])
        for i, (s, l) in enumerate(pos_plot):
            w.writerow([f"P{i+1}", "positive", s, f"{vals_p[i]:.6f}"])
        for i, (s, l) in enumerate(neg_plot):
            w.writerow([f"N{i+1}", "negative", s, f"{vals_n[i]:.6f}"])
    print(f"CSV: {csv_path}")

for tag, sents_sel, sents_plt in [("pos", pos_select, pos_plot), ("neg", neg_select, neg_plot)]:
    csv_path = os.path.join(SCRIPT_DIR, f"filtered_sentences_gemma_{tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["idx", "usage", "sentence", "ground_truth_label"])
        for i, (s, l) in enumerate(sents_sel):
            w.writerow([i+1, "select", s, l])
        for i, (s, l) in enumerate(sents_plt):
            w.writerow([len(sents_sel)+i+1, "plot", s, l])
    print(f"CSV: {csv_path}")

print(f"\n{'='*70}")
print(f"VALUES: Avg Top-5 Positive Features (Gemma-2-2B)")
print(f"Features: {list(top_pos_idx[:5])}")
print(f"Pos mean={pos_avg.mean():.4f}, Neg mean={neg_avg.mean():.4f}")
print(f"\nVALUES: Avg Top-5 Negative Features (Gemma-2-2B)")
print(f"Features: {list(top_neg_idx[:5])}")
print(f"Pos mean={neg_avg_n.mean():.4f}, Neg mean={neg_avg_nn.mean():.4f}")
print(f"{'='*70}")

print("\n\nDone.")
