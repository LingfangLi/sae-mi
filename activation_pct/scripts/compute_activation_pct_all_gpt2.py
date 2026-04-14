# -*- coding: utf-8 -*-
"""Per-sample activation percentage for GPT-2 Small on all 3 tasks
SAE: gpt2-small-hook-z-kk (attention hook_z)
Uses TransformerLens + SAELens
"""

import os, random, gc
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

# ── Load Model ────────────────────────────────────────────────
from transformer_lens import HookedTransformer
from sae_lens import SAE

print("Loading GPT-2 Small (TransformerLens)...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.eval()

RELEASE = "gpt2-small-hook-z-kk"

# ── Task configs ──────────────────────────────────────────────
TASKS = {
    "sst2": {"layer": 7, "name": "SST-2 (Sentiment)"},
    "mrpc": {"layer": 2, "name": "MRPC (Paraphrase)"},
    "mt":   {"layer": 8, "name": "MT (Europarl)"},
}

# ── Load all datasets ─────────────────────────────────────────
from datasets import load_dataset

# SST-2
print("Loading SST-2...")
ds_sst2 = load_dataset("glue", "sst2")
sst2_val = ds_sst2["validation"]
sst2_train = ds_sst2["train"]
sst2_sents = [ex["sentence"] for ex in sst2_val]
remaining = 1000 - len(sst2_sents)
if remaining > 0:
    for i in range(remaining):
        sst2_sents.append(sst2_train[i]["sentence"])
print(f"  SST-2: {len(sst2_sents)} samples")

# MRPC
print("Loading MRPC...")
ds_mrpc = load_dataset("glue", "mrpc")
mrpc_val = ds_mrpc["validation"]
mrpc_train = ds_mrpc["train"]
mrpc_sents = [f"{ex['sentence1']} <|endoftext|> {ex['sentence2']}" for ex in mrpc_val]
remaining = 1000 - len(mrpc_sents)
if remaining > 0:
    for i in range(remaining):
        ex = mrpc_train[i]
        mrpc_sents.append(f"{ex['sentence1']} <|endoftext|> {ex['sentence2']}")
print(f"  MRPC: {len(mrpc_sents)} samples")

# MT (Europarl)
print("Loading Europarl...")
DATA_DIR = "/mnt/scratch/users/yangwr/Lingfang/saes-mi/data"
with open(os.path.join(DATA_DIR, "europarl-v7.fr-en.en"), 'r') as f:
    en_lines = [l.strip() for l in f if l.strip()][:10000]
with open(os.path.join(DATA_DIR, "europarl-v7.fr-en.fr"), 'r') as f:
    fr_lines = [l.strip() for l in f if l.strip()][:10000]
random.seed(42)
N_VAL = 500
pos_idx = random.sample(range(len(en_lines)), N_VAL)
neg_idx_en = random.sample(range(len(en_lines)), N_VAL)
neg_idx_fr = random.sample(range(len(fr_lines)), N_VAL)
mt_sents = []
for i in pos_idx:
    mt_sents.append(f"{en_lines[i]} <|endoftext|> {fr_lines[i]}")
for i in range(N_VAL):
    mt_sents.append(f"{en_lines[neg_idx_en[i]]} <|endoftext|> {fr_lines[neg_idx_fr[i]]}")
print(f"  MT: {len(mt_sents)} samples")

all_sentences = {"sst2": sst2_sents, "mrpc": mrpc_sents, "mt": mt_sents}

# ── Helper: compute per-sample stats ─────────────────────────
def compute_per_sample(task_key, layer, sentences):
    hook_name = f"blocks.{layer}.attn.hook_z"
    sae_id = f"blocks.{layer}.hook_z"

    print(f"\nLoading SAE: {RELEASE} / {sae_id}")
    try:
        sae = SAE.from_pretrained(RELEASE, sae_id)
    except:
        sae = SAE.from_pretrained(RELEASE, sae_id)[0]
    sae.to(device).eval()
    num_latents = sae.cfg.d_sae

    print(f"SAE: layer={layer}, d_sae={num_latents}")
    print(f"Computing per-sample activation for {len(sentences)} samples...")

    all_pcts = []
    all_active_counts = []
    all_mean_vals = []
    all_pooled = []

    with torch.no_grad():
        for idx in range(len(sentences)):
            tokens = model.to_tokens(sentences[idx])
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            hook_acts = cache[hook_name]
            sae_acts = sae.encode(hook_acts)
            pooled = sae_acts.mean(dim=1).cpu().numpy().flatten()

            active_count = (pooled > 0).sum()
            pct = active_count / num_latents * 100
            all_pcts.append(pct)
            all_active_counts.append(active_count)
            nz = pooled[pooled > 0]
            all_mean_vals.append(nz.mean() if len(nz) > 0 else 0)
            all_pooled.append(pooled)

            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(sentences)}, avg pct: {np.mean(all_pcts):.2f}%")

    del sae
    gc.collect()
    torch.cuda.empty_cache()

    return (np.array(all_pcts), np.array(all_active_counts),
            np.array(all_mean_vals), np.array(all_pooled), num_latents)

# ── Helper: threshold sweep + print + plot + save ─────────────
thresholds = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

def report_and_plot(task_key, task_name, layer, all_pcts, all_active_counts,
                    all_mean_vals, all_pooled, num_latents):

    print(f"\n{'='*70}")
    print(f"RESULTS: {task_name} | GPT-2 Small | Layer {layer} | d_sae={num_latents}")
    print(f"{'='*70}")
    print(f"Samples: {len(all_pcts)}")
    print(f"Active features per sample:")
    print(f"  Mean:   {all_active_counts.mean():.1f} ({all_pcts.mean():.2f}%)")
    print(f"  Median: {np.median(all_active_counts):.1f} ({np.median(all_pcts):.2f}%)")
    print(f"  Min:    {all_active_counts.min()} ({all_pcts.min():.2f}%)")
    print(f"  Max:    {all_active_counts.max()} ({all_pcts.max():.2f}%)")
    print(f"  Std:    {all_active_counts.std():.1f} ({all_pcts.std():.2f}%)")
    print(f"Mean non-zero value: {all_mean_vals.mean():.6f}")

    print(f"\nThreshold sweep:")
    print(f"{'Threshold':<12} {'Mean Active':<15} {'Mean %':<10} {'Median %':<10} {'Max %':<10}")
    print("-" * 60)

    sweep = []
    for thr in thresholds:
        p = np.array([(row > thr).sum() / num_latents * 100 for row in all_pooled])
        c = np.array([(row > thr).sum() for row in all_pooled])
        sweep.append((thr, c.mean(), p.mean(), np.median(p), p.max()))
        print(f"{thr:<12.3f} {c.mean():<15.1f} {p.mean():<10.2f} {np.median(p):<10.2f} {p.max():<10.2f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.hist(all_pcts, bins=50, color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(all_pcts.mean(), color='black', linestyle='--', linewidth=2,
               label=f'Mean={all_pcts.mean():.2f}%')
    ax.set_xlabel('% of features activated (threshold=0)')
    ax.set_ylabel('Count (samples)')
    ax.set_title(f'(A) Per-Sample Activation %\nGPT-2 Small, Layer {layer}, {task_name}')
    ax.legend()

    ax = axes[0, 1]
    all_nz = all_pooled[all_pooled > 0]
    if len(all_nz) > 0:
        ax.hist(all_nz, bins=200, color='#3498db', edgecolor='black', linewidth=0.2, alpha=0.8,
                range=(0, np.percentile(all_nz, 99)))
        ax.axvline(np.median(all_nz), color='red', linestyle='--', linewidth=2,
                   label=f'Median={np.median(all_nz):.4f}')
    ax.set_xlabel('Activation value (non-zero, mean-pooled)')
    ax.set_ylabel('Count')
    ax.set_title('(B) Distribution of Non-Zero Activation Values')
    ax.legend()

    ax = axes[1, 0]
    thr_vals = [r[0] for r in sweep]
    mean_pcts = [r[2] for r in sweep]
    ax.plot(thr_vals, mean_pcts, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    for t, p in zip(thr_vals, mean_pcts):
        ax.annotate(f'{p:.1f}%', (t, p), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Mean % activated features')
    ax.set_title('(C) Effect of Threshold on Activation %')
    ax.set_xscale('symlog', linthresh=0.001)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for thr in [0, 0.01, 0.05, 0.1]:
        p = np.array([(row > thr).sum() / num_latents * 100 for row in all_pooled])
        ax.hist(p, bins=40, alpha=0.5, label=f'thr={thr}', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('% of features activated per sample')
    ax.set_ylabel('Count (samples)')
    ax.set_title('(D) Per-Sample Activation % at Different Thresholds')
    ax.legend()

    plt.tight_layout()
    base = os.path.join(SCRIPT_DIR, f"{task_key}_gpt2_activation_pct")
    plt.savefig(f"{base}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{base}.pdf", bbox_inches='tight')
    plt.close()
    print(f"\nPlots: {base}.png / .pdf")

    with open(f"{base}.txt", 'w') as f:
        f.write(f"Per-Sample Activated Feature Percentage\n")
        f.write(f"Model: GPT-2 Small\nSAE: {RELEASE} (attention hook_z)\n")
        f.write(f"Layer: {layer}\nTask: {task_name}\nd_sae: {num_latents}\nSamples: {len(all_pcts)}\n\n")
        f.write(f"{'Threshold':<12} {'Mean Count':<15} {'Mean %':<10} {'Median %':<10} {'Max %':<10}\n")
        f.write("-" * 60 + "\n")
        for thr, cnt, mp, mdp, mxp in sweep:
            f.write(f"{thr:<12.3f} {cnt:<15.1f} {mp:<10.2f} {mdp:<10.2f} {mxp:<10.2f}\n")
    print(f"Text: {base}.txt")

# ── Run all tasks ─────────────────────────────────────────────
for task_key, cfg in TASKS.items():
    print(f"\n{'#'*70}")
    print(f"# {cfg['name']} (Layer {cfg['layer']})")
    print(f"{'#'*70}")
    pcts, counts, vals, pooled, n_lat = compute_per_sample(
        task_key, cfg['layer'], all_sentences[task_key])
    report_and_plot(task_key, cfg['name'], cfg['layer'],
                    pcts, counts, vals, pooled, n_lat)

print("\n\nAll done.")
