# -*- coding: utf-8 -*-
"""Per-sample activation percentage for DeepSeek on MRPC

For every input:
  1. Get SAE activations (mean-pooled over tokens)
  2. Count features with value > threshold
  3. Compute percentage = count / total_features

Also sweep thresholds to show how percentage drops.
"""

import os
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

# ── Load MRPC ─────────────────────────────────────────────────
from datasets import load_dataset

print("Loading MRPC dataset...")
dataset = load_dataset("glue", "mrpc")
val_dataset = dataset["validation"]   # 408 samples
train_dataset = dataset["train"]      # 3668 samples

print(f"Val: {len(val_dataset)}, Train: {len(train_dataset)}")

# ── Load Model + SAE ──────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
)
model.eval()

# Build sentences: sentence1 + EOS + sentence2
sentences = []
labels = []
for ex in val_dataset:
    sentences.append(f"{ex['sentence1']} {EOS} {ex['sentence2']}")
    labels.append(ex["label"])
# Add from train to reach 1000
remaining = 1000 - len(sentences)
if remaining > 0:
    for i in range(remaining):
        ex = train_dataset[i]
        sentences.append(f"{ex['sentence1']} {EOS} {ex['sentence2']}")
        labels.append(ex["label"])

print(f"Total samples: {len(sentences)}")

LAYER = 14  # best probing layer for MRPC
sae = Sae.load_from_hub("EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k",
                         hookpoint=f"layers.{LAYER}.mlp")
sae.to(device).eval()
num_latents = sae.num_latents
K = sae.cfg.k
print(f"SAE: layer={LAYER}, num_latents={num_latents}, TopK={K}")

# ── Hook for MLP output ───────────────────────────────────────
captured = {}
def hook_fn(module, inp, out):
    captured['out'] = out
hook = model.model.layers[LAYER].mlp.register_forward_hook(hook_fn)

# ── Compute per-sample stats ──────────────────────────────────
print("\nComputing per-sample activation percentages...")

all_pcts = []
all_active_counts = []
all_mean_vals = []
all_pooled = []

with torch.no_grad():
    for idx in range(len(sentences)):
        sent = sentences[idx]
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
        model(**enc, use_cache=False)
        mlp_out = captured['out']
        seq_len = mlp_out.shape[1]
        flat = mlp_out.view(-1, mlp_out.shape[-1])

        enc_output = sae.encode(flat)
        dense = torch.zeros(seq_len, num_latents, device=device, dtype=torch.float32)
        dense.scatter_(1, enc_output.top_indices.long(), enc_output.top_acts.float())

        # Mean pool over sequence
        pooled = dense.mean(dim=0).cpu().numpy()

        active_count = (pooled > 0).sum()
        pct = active_count / num_latents * 100

        all_pcts.append(pct)
        all_active_counts.append(active_count)
        nz = pooled[pooled > 0]
        all_mean_vals.append(nz.mean() if len(nz) > 0 else 0)
        all_pooled.append(pooled)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(sentences)}, avg pct so far: {np.mean(all_pcts):.2f}%")

hook.remove()

all_pcts = np.array(all_pcts)
all_active_counts = np.array(all_active_counts)
all_mean_vals = np.array(all_mean_vals)
all_pooled = np.array(all_pooled)

print(f"\n{'='*70}")
print(f"RESULTS: Per-Sample Activated Feature Percentage (threshold=0)")
print(f"{'='*70}")
print(f"Task: MRPC (Paraphrase Detection)")
print(f"Total features: {num_latents}")
print(f"TopK: {K}")
print(f"Samples: {len(sentences)}")
print(f"")
print(f"Active features per sample:")
print(f"  Mean:   {all_active_counts.mean():.1f} ({all_pcts.mean():.2f}%)")
print(f"  Median: {np.median(all_active_counts):.1f} ({np.median(all_pcts):.2f}%)")
print(f"  Min:    {all_active_counts.min()} ({all_pcts.min():.2f}%)")
print(f"  Max:    {all_active_counts.max()} ({all_pcts.max():.2f}%)")
print(f"  Std:    {all_active_counts.std():.1f} ({all_pcts.std():.2f}%)")
print(f"")
print(f"Mean non-zero activation value per sample:")
print(f"  Mean:   {all_mean_vals.mean():.6f}")
print(f"  Median: {np.median(all_mean_vals):.6f}")

# ── Threshold sweep ───────────────────────────────────────────
print(f"\n{'='*70}")
print(f"THRESHOLD SWEEP: How percentage changes with threshold")
print(f"{'='*70}")

thresholds = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
sweep_results = []

print(f"{'Threshold':<12} {'Mean Active':<15} {'Mean %':<10} {'Median %':<10} {'Max %':<10}")
print("-" * 60)

for thr in thresholds:
    pcts = [(row > thr).sum() / num_latents * 100 for row in all_pooled]
    pcts = np.array(pcts)
    counts = [(row > thr).sum() for row in all_pooled]
    counts = np.array(counts)
    sweep_results.append((thr, counts.mean(), pcts.mean(), np.median(pcts), pcts.max()))
    print(f"{thr:<12.3f} {counts.mean():<15.1f} {pcts.mean():<10.2f} {np.median(pcts):<10.2f} {pcts.max():<10.2f}")

# ── Plots ─────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
ax.hist(all_pcts, bins=50, color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
ax.axvline(all_pcts.mean(), color='black', linestyle='--', linewidth=2,
           label=f'Mean={all_pcts.mean():.2f}%')
ax.set_xlabel('% of features activated (threshold=0)', fontsize=11)
ax.set_ylabel('Count (samples)', fontsize=11)
ax.set_title(f'(A) Per-Sample Activation % (threshold=0)\nDeepSeek TopK={K}, Layer {LAYER}, MRPC', fontsize=11)
ax.legend(fontsize=10)

ax = axes[0, 1]
all_nz = all_pooled[all_pooled > 0]
ax.hist(all_nz, bins=200, color='#3498db', edgecolor='black', linewidth=0.2, alpha=0.8,
        range=(0, np.percentile(all_nz, 99)))
ax.axvline(np.median(all_nz), color='red', linestyle='--', linewidth=2,
           label=f'Median={np.median(all_nz):.4f}')
ax.set_xlabel('Activation value (non-zero, mean-pooled)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('(B) Distribution of Non-Zero Activation Values\n(after mean pooling over tokens)', fontsize=11)
ax.legend(fontsize=10)

ax = axes[1, 0]
thr_vals = [r[0] for r in sweep_results]
mean_pcts = [r[2] for r in sweep_results]
ax.plot(thr_vals, mean_pcts, 'o-', color='#e74c3c', linewidth=2, markersize=8)
for t, p in zip(thr_vals, mean_pcts):
    ax.annotate(f'{p:.1f}%', (t, p), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8)
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Mean % activated features', fontsize=11)
ax.set_title('(C) Effect of Threshold on Activation %\n(higher threshold -> fewer "active" features)', fontsize=11)
ax.set_xscale('symlog', linthresh=0.001)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
for thr in [0, 0.01, 0.05, 0.1]:
    pcts = np.array([(row > thr).sum() / num_latents * 100 for row in all_pooled])
    ax.hist(pcts, bins=40, alpha=0.5, label=f'thr={thr}', edgecolor='black', linewidth=0.3)
ax.set_xlabel('% of features activated per sample', fontsize=11)
ax.set_ylabel('Count (samples)', fontsize=11)
ax.set_title('(D) Per-Sample Activation % at Different Thresholds', fontsize=11)
ax.legend(fontsize=9)

plt.tight_layout()
out_png = os.path.join(SCRIPT_DIR, "mrpc_deepseek_activation_pct.png")
out_pdf = os.path.join(SCRIPT_DIR, "mrpc_deepseek_activation_pct.pdf")
plt.savefig(out_png, dpi=150, bbox_inches='tight')
plt.savefig(out_pdf, bbox_inches='tight')
print(f"\nPlots saved: {out_png}")
print(f"             {out_pdf}")

# ── Save text summary ─────────────────────────────────────────
txt_path = os.path.join(SCRIPT_DIR, "mrpc_deepseek_activation_pct.txt")
with open(txt_path, 'w') as f:
    f.write("Per-Sample Activated Feature Percentage\n")
    f.write(f"Model: DeepSeek-R1-Distill-Qwen-1.5B\n")
    f.write(f"SAE: EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k (TopK={K})\n")
    f.write(f"Layer: {LAYER}\n")
    f.write(f"Task: MRPC (Paraphrase Detection)\n")
    f.write(f"Total features: {num_latents}\n")
    f.write(f"Samples: {len(sentences)}\n\n")
    f.write(f"{'Threshold':<12} {'Mean Count':<15} {'Mean %':<10} {'Median %':<10} {'Max %':<10}\n")
    f.write("-" * 60 + "\n")
    for thr, cnt, mp, mdp, mxp in sweep_results:
        f.write(f"{thr:<12.3f} {cnt:<15.1f} {mp:<10.2f} {mdp:<10.2f} {mxp:<10.2f}\n")
print(f"Text saved: {txt_path}")
print("\nDone.")
