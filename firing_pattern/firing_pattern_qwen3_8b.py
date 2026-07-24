# -*- coding: utf-8 -*-
"""Firing pattern visualization for Qwen3-8B-Base on SST-2 (Fig 4 support)

Adapted from firing_pattern_qwen3.py (Qwen3-0.6B version):
  - Model:  Qwen/Qwen3-8B-Base
  - SAE:    qwen-scope-3-8b-base-w64k-l100  (residual, d_sae=65536)
  - sae_id: "layer{N}"  (no underscore — different from mwhanna's "layer_{N}")
  - Pooling: MAX-pool over sequence (paper §3.2)
  - LAYER passed via --layer CLI (set to Stage-1 best layer)
"""

import os, random, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=int, required=True,
                    help="Best SST-2 layer for Qwen3-8B (from probing)")
parser.add_argument("--n_select", type=int, default=200,
                    help="N sentences (per polarity) to use for feature selection")
parser.add_argument("--n_plot", type=int, default=30,
                    help="N sentences (per polarity) to display in bar plot")
args = parser.parse_args()

LAYER = args.layer
N_SELECT = args.n_select
N_PLOT = args.n_plot
MODEL_NAME = "Qwen3-8B"

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "qwen3_8b")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', CACHE_DIR)
os.environ.setdefault('TRANSFORMERS_CACHE', CACHE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load SST-2 ────────────────────────────────────────────────
from datasets import load_dataset

print("Loading SST-2...")
ds = load_dataset("nyu-mll/glue", "sst2", cache_dir=CACHE_DIR)
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

pos_candidates = pos_sents
neg_candidates = neg_sents
print(f"Candidates: {len(pos_candidates)} pos, {len(neg_candidates)} neg")

# ── Load Model + SAE ──────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

base_model_id = "Qwen/Qwen3-8B-Base"
print(f"Loading {base_model_id} (bf16)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
)
model.eval()

release = "qwen-scope-3-8b-base-w64k-l100"
sae_id = f"layer{LAYER}"                          # no underscore for qwen-scope
print(f"Loading SAE: {release}, {sae_id}")
loaded = SAE.from_pretrained(release, sae_id)
sae = loaded[0] if isinstance(loaded, tuple) else loaded
sae.to(device).eval()
d_sae = sae.cfg.d_sae
print(f"SAE: layer={LAYER}, d_sae={d_sae}, k={getattr(sae.cfg,'k','?')}")

# ── Helper: get layer residual ────────────────────────────────
def get_layer_rep(text, layer_idx):
    enc = tokenizer(text, return_tensors="pt", padding=False,
                    truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx + 1]        # (1, seq, d_model)


# ── Zero-shot sentiment prediction (same prompt as pretrained script) ──
def predict_sentiment_zeroshot(sentence):
    prompt = f"""Classify the sentiment as positive or negative.

Sentence: {sentence}
Sentiment:"""
    enc = tokenizer(prompt, return_tensors="pt", padding=False,
                    truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    next_logits = out.logits[0, -1, :]
    next_id = torch.argmax(next_logits).item()
    tok = tokenizer.decode([next_id]).strip().lower()
    if "positive" in tok or tok.startswith("pos"):
        return 1
    if "negative" in tok or tok.startswith("neg"):
        return 0
    pos_id = tokenizer.encode(" positive", add_special_tokens=False)[-1]
    neg_id = tokenizer.encode(" negative", add_special_tokens=False)[-1]
    return 1 if next_logits[pos_id] > next_logits[neg_id] else 0


# ── Filter sentences by zero-shot correctness ─────────────────
print("\nStep 0: Filtering sentences by zero-shot prediction...")

def collect(candidates, target_label, tag):
    select, plot, plot_done = [], [], False
    for i, (s, l) in enumerate(candidates):
        pred = predict_sentiment_zeroshot(s)
        if pred == target_label:
            if len(select) < N_SELECT:
                select.append((s, l))
            elif not plot_done and len(plot) < N_PLOT:
                plot.append((s, l))
                if len(plot) >= N_PLOT:
                    plot_done = True
        if (i + 1) % 20 == 0:
            print(f"  [{tag}] Checked {i+1}/{len(candidates)}: {len(select)} select, {len(plot)} plot")
        if len(select) >= N_SELECT and plot_done:
            break
    if len(plot) < N_PLOT:
        extra = N_PLOT - len(plot)
        plot.extend(select[:extra])
        print(f"  [{tag}] borrowed {extra} from select for plot")
    return select, plot

print("  --- Positive sentences ---")
pos_select, pos_plot = collect(pos_candidates, target_label=1, tag="pos")
print("  --- Negative sentences ---")
neg_select, neg_plot = collect(neg_candidates, target_label=0, tag="neg")
print(f"After filter: pos_select={len(pos_select)}, pos_plot={len(pos_plot)}, "
      f"neg_select={len(neg_select)}, neg_plot={len(neg_plot)}")

# ── Extract SAE activations (MAX-POOL) ────────────────────────
def get_sae_activation(sentence):
    layer_h = get_layer_rep(sentence, LAYER)               # (1, seq, d_model)
    sae_feats = sae.encode(layer_h.to(sae.dtype))          # (1, seq, d_sae)
    # MAX-pool over sequence (paper §3.2)
    pooled = sae_feats.max(dim=1).values[0].to(torch.float32).detach().cpu().numpy()
    return pooled

print(f"\nStep 1: Extracting activations ({len(pos_select)} pos, {len(neg_select)} neg)...")
pos_acts = np.array([get_sae_activation(s) for s, _ in pos_select])
neg_acts = np.array([get_sae_activation(s) for s, _ in neg_select])

# ── Feature selection ─────────────────────────────────────────
print("\nStep 2: Identifying task-relevant features (positive_mean - negative_mean)...")
pos_mean = pos_acts.mean(axis=0)
neg_mean = neg_acts.mean(axis=0)
diff = pos_mean - neg_mean

top_pos_idx = np.argsort(diff)[::-1][:10]              # features most POS - NEG
top_neg_idx = np.argsort(diff)[:10]                    # features most NEG - POS

print("Top 10 positive-responsible features (idx | pos_mean | neg_mean | diff):")
for rank, fi in enumerate(top_pos_idx):
    print(f"  #{rank+1}: F{fi} | {pos_mean[fi]:.4f} | {neg_mean[fi]:.4f} | {diff[fi]:+.4f}")
print("Top 10 negative-responsible features:")
for rank, fi in enumerate(top_neg_idx):
    print(f"  #{rank+1}: F{fi} | {pos_mean[fi]:.4f} | {neg_mean[fi]:.4f} | {diff[fi]:+.4f}")

# ── Extract plot activations ──────────────────────────────────
print("\nStep 3: Extracting activations for plot sentences...")
pos_plot_acts = np.array([get_sae_activation(s) for s, _ in pos_plot])
neg_plot_acts = np.array([get_sae_activation(s) for s, _ in neg_plot])


def plot_avg_features(feat_indices, sentiment_type, pos_acts_mat, neg_acts_mat, model_name):
    pos_avg = pos_acts_mat[:, feat_indices].mean(axis=1)
    neg_avg = neg_acts_mat[:, feat_indices].mean(axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    ymax = max(pos_avg.max(), neg_avg.max()) * 1.15 or 1
    x_pos = np.arange(len(pos_avg))
    x_neg = np.arange(len(neg_avg))

    ax1.bar(x_pos, pos_avg, color='#2ecc71', edgecolor='black', linewidth=0.5, width=0.8)
    ax1.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax1.set_title(f'{model_name}: Avg Top-5 {sentiment_type}-responsible features on Positive Sentences', fontsize=13)
    ax1.set_ylim(0, ymax)
    ax1.axhline(y=pos_avg.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={pos_avg.mean():.4f}')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'P{i+1}' for i in range(len(pos_avg))], fontsize=8, rotation=45)
    ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x_neg, neg_avg, color='#e74c3c', edgecolor='black', linewidth=0.5, width=0.8)
    ax2.set_ylabel('Avg Activation Intensity', fontsize=12)
    ax2.set_title(f'{model_name}: Avg Top-5 {sentiment_type}-responsible features on Negative Sentences', fontsize=13)
    ax2.set_ylim(0, ymax)
    ax2.axhline(y=neg_avg.mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={neg_avg.mean():.4f}')
    ax2.set_xticks(x_neg)
    ax2.set_xticklabels([f'N{i+1}' for i in range(len(neg_avg))], fontsize=8, rotation=45)
    ax2.set_xlabel('Sentences', fontsize=12)
    ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig, pos_avg, neg_avg


print("\nStep 4: Generating plots (top-5 pos + top-5 neg features)...")
fig, pos_avg, neg_avg = plot_avg_features(top_pos_idx[:5], "Positive", pos_plot_acts, neg_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_qwen3_8b_pos_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

fig, neg_avg_n, neg_avg_nn = plot_avg_features(top_neg_idx[:5], "Negative", pos_plot_acts, neg_plot_acts, MODEL_NAME)
out = os.path.join(OUT_DIR, "firing_pattern_qwen3_8b_neg_avg5.png")
fig.savefig(out, dpi=150, bbox_inches='tight')
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Save CSVs (per-sentence firing values + which sentences chosen) ──
import csv
for tag, feat_ids, label in [("pos", top_pos_idx[:5], "avg_top5_pos_features"),
                              ("neg", top_neg_idx[:5], "avg_top5_neg_features")]:
    vals_p = pos_plot_acts[:, feat_ids].mean(axis=1)
    vals_n = neg_plot_acts[:, feat_ids].mean(axis=1)
    csv_path = os.path.join(OUT_DIR, f"firing_values_qwen3_8b_{tag}_avg5.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["sentence_id", "sentence_type", "sentence_text", label])
        for i, (s, l) in enumerate(pos_plot):
            w.writerow([f"P{i+1}", "positive", s, f"{vals_p[i]:.6f}"])
        for i, (s, l) in enumerate(neg_plot):
            w.writerow([f"N{i+1}", "negative", s, f"{vals_n[i]:.6f}"])
    print(f"CSV: {csv_path}")

for tag, sents_sel, sents_plt in [("pos", pos_select, pos_plot),
                                    ("neg", neg_select, neg_plot)]:
    csv_path = os.path.join(OUT_DIR, f"filtered_sentences_qwen3_8b_{tag}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["idx", "usage", "sentence", "ground_truth_label"])
        for i, (s, l) in enumerate(sents_sel):
            w.writerow([i+1, "select", s, l])
        for i, (s, l) in enumerate(sents_plt):
            w.writerow([len(sents_sel)+i+1, "plot", s, l])
    print(f"CSV: {csv_path}")

# ── Save top-feature IDs and pos/neg means for teacher's plotting ──
np.savez(
    os.path.join(OUT_DIR, f"top_features_qwen3_8b_L{LAYER}.npz"),
    top_pos_idx=top_pos_idx.astype(np.int32),
    top_neg_idx=top_neg_idx.astype(np.int32),
    pos_mean=pos_mean.astype(np.float32),
    neg_mean=neg_mean.astype(np.float32),
    diff=diff.astype(np.float32),
    layer=LAYER, d_sae=d_sae,
)

print(f"\n{'='*70}")
print(f"VALUES: Avg Top-5 Positive Features ({MODEL_NAME}, L{LAYER})")
print(f"Features: {list(top_pos_idx[:5])}")
print(f"Pos mean={pos_avg.mean():.4f}, Neg mean={neg_avg.mean():.4f}")
print(f"\nVALUES: Avg Top-5 Negative Features ({MODEL_NAME}, L{LAYER})")
print(f"Features: {list(top_neg_idx[:5])}")
print(f"Pos mean={neg_avg_n.mean():.4f}, Neg mean={neg_avg_nn.mean():.4f}")
print(f"{'='*70}\nDone.")
