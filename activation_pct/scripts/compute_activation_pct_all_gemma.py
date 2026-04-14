# -*- coding: utf-8 -*-
"""Per-sample activation percentage for Gemma-2-2B on all 3 tasks
SAE: gemma-scope-2b-pt-att-canonical (attention SAE, 16k)
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

print("Loading Gemma-2-2B (TransformerLens)...")
model = HookedTransformer.from_pretrained("google/gemma-2b", device=device)
model.eval()

RELEASE = "gemma-scope-2b-pt-att-canonical"

TASKS = {
    "sst2": {"layer": 12, "name": "SST-2 (Sentiment)"},
    "mrpc": {"layer": 12, "name": "MRPC (Paraphrase)"},
    "mt":   {"layer": 7,  "name": "MT (Europarl)"},
}

# ── Load all datasets ─────────────────────────────────────────
from datasets import load_dataset

EOS = "<|endoftext|>"

print("Loading SST-2...")
ds_sst2 = load_dataset("glue", "sst2")
sst2_sents = [ex["sentence"] for ex in ds_sst2["validation"]]
remaining = 1000 - len(sst2_sents)
if remaining > 0:
    for i in range(remaining):
        sst2_sents.append(ds_sst2["train"][i]["sentence"])
print(f"  SST-2: {len(sst2_sents)} samples")

print("Loading MRPC...")
ds_mrpc = load_dataset("glue", "mrpc")
mrpc_sents = [f"{ex['sentence1']} {EOS} {ex['sentence2']}" for ex in ds_mrpc["validation"]]
remaining = 1000 - len(mrpc_sents)
if remaining > 0:
    for i in range(remaining):
        ex = ds_mrpc["train"][i]
        mrpc_sents.append(f"{ex['sentence1']} {EOS} {ex['sentence2']}")
print(f"  MRPC: {len(mrpc_sents)} samples")

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
    mt_sents.append(f"{en_lines[i]} {EOS} {fr_lines[i]}")
for i in range(N_VAL):
    mt_sents.append(f"{en_lines[neg_idx_en[i]]} {EOS} {fr_lines[neg_idx_fr[i]]}")
print(f"  MT: {len(mt_sents)} samples")

all_sentences = {"sst2": sst2_sents, "mrpc": mrpc_sents, "mt": mt_sents}

# ── Compute per-sample ────────────────────────────────────────
def compute_per_sample(task_key, layer, sentences):
    hook_name = f"blocks.{layer}.attn.hook_z"
    sae_id = f"layer_{layer}/width_16k/canonical"

    print(f"\nLoading SAE: {RELEASE} / {sae_id}")
    try:
        sae = SAE.from_pretrained(RELEASE, sae_id)
    except:
        sae = SAE.from_pretrained(RELEASE, sae_id)[0]
    sae.to(device).eval()
    num_latents = sae.cfg.d_sae
    print(f"SAE: layer={layer}, d_sae={num_latents}")

    all_pcts, all_counts, all_vals, all_pooled = [], [], [], []
    with torch.no_grad():
        for idx in range(len(sentences)):
            tokens = model.to_tokens(sentences[idx], prepend_bos=True)
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            hook_acts = cache[hook_name]
            sae_acts = sae.encode(hook_acts)
            pooled = sae_acts.mean(dim=1)[0].cpu().numpy().flatten()

            ac = (pooled > 0).sum()
            all_pcts.append(ac / num_latents * 100)
            all_counts.append(ac)
            nz = pooled[pooled > 0]
            all_vals.append(nz.mean() if len(nz) > 0 else 0)
            all_pooled.append(pooled)
            if (idx + 1) % 200 == 0:
                print(f"  {idx+1}/{len(sentences)}, avg pct: {np.mean(all_pcts):.2f}%")

    del sae; gc.collect(); torch.cuda.empty_cache()
    return (np.array(all_pcts), np.array(all_counts),
            np.array(all_vals), np.array(all_pooled), num_latents)

# ── Report + plot ─────────────────────────────────────────────
thresholds = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

def report_and_plot(task_key, task_name, layer, all_pcts, all_counts,
                    all_vals, all_pooled, num_latents):
    print(f"\n{'='*70}")
    print(f"RESULTS: {task_name} | Gemma-2-2B | Layer {layer} | d_sae={num_latents}")
    print(f"{'='*70}")
    print(f"Samples: {len(all_pcts)}")
    print(f"Active features per sample:")
    print(f"  Mean:   {all_counts.mean():.1f} ({all_pcts.mean():.2f}%)")
    print(f"  Median: {np.median(all_counts):.1f} ({np.median(all_pcts):.2f}%)")
    print(f"  Min:    {all_counts.min()} ({all_pcts.min():.2f}%)")
    print(f"  Max:    {all_counts.max()} ({all_pcts.max():.2f}%)")
    print(f"  Std:    {all_counts.std():.1f} ({all_pcts.std():.2f}%)")
    print(f"Mean non-zero value: {all_vals.mean():.6f}")

    sweep = []
    print(f"\n{'Threshold':<12} {'Mean Active':<15} {'Mean %':<10} {'Median %':<10} {'Max %':<10}")
    print("-" * 60)
    for thr in thresholds:
        p = np.array([(row > thr).sum() / num_latents * 100 for row in all_pooled])
        c = np.array([(row > thr).sum() for row in all_pooled])
        sweep.append((thr, c.mean(), p.mean(), np.median(p), p.max()))
        print(f"{thr:<12.3f} {c.mean():<15.1f} {p.mean():<10.2f} {np.median(p):<10.2f} {p.max():<10.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    ax.hist(all_pcts, bins=50, color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axvline(all_pcts.mean(), color='black', linestyle='--', linewidth=2, label=f'Mean={all_pcts.mean():.2f}%')
    ax.set_xlabel('% of features activated (threshold=0)'); ax.set_ylabel('Count (samples)')
    ax.set_title(f'(A) Per-Sample Activation %\nGemma-2-2B, Layer {layer}, {task_name}'); ax.legend()

    ax = axes[0, 1]
    all_nz = all_pooled[all_pooled > 0]
    if len(all_nz) > 0:
        ax.hist(all_nz, bins=200, color='#3498db', edgecolor='black', linewidth=0.2, alpha=0.8,
                range=(0, np.percentile(all_nz, 99)))
        ax.axvline(np.median(all_nz), color='red', linestyle='--', linewidth=2, label=f'Median={np.median(all_nz):.4f}')
    ax.set_xlabel('Activation value (non-zero, mean-pooled)'); ax.set_ylabel('Count')
    ax.set_title('(B) Distribution of Non-Zero Activation Values'); ax.legend()

    ax = axes[1, 0]
    ax.plot([r[0] for r in sweep], [r[2] for r in sweep], 'o-', color='#e74c3c', linewidth=2, markersize=8)
    for t, p in zip([r[0] for r in sweep], [r[2] for r in sweep]):
        ax.annotate(f'{p:.1f}%', (t, p), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    ax.set_xlabel('Threshold'); ax.set_ylabel('Mean % activated features')
    ax.set_title('(C) Effect of Threshold on Activation %')
    ax.set_xscale('symlog', linthresh=0.001); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for thr in [0, 0.01, 0.05, 0.1]:
        p = np.array([(row > thr).sum() / num_latents * 100 for row in all_pooled])
        ax.hist(p, bins=40, alpha=0.5, label=f'thr={thr}', edgecolor='black', linewidth=0.3)
    ax.set_xlabel('% of features activated per sample'); ax.set_ylabel('Count (samples)')
    ax.set_title('(D) Per-Sample Activation % at Different Thresholds'); ax.legend()

    plt.tight_layout()
    base = os.path.join(SCRIPT_DIR, f"{task_key}_gemma_activation_pct")
    plt.savefig(f"{base}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{base}.pdf", bbox_inches='tight')
    plt.close()
    print(f"\nPlots: {base}.png / .pdf")

    with open(f"{base}.txt", 'w') as f:
        f.write(f"Per-Sample Activated Feature Percentage\nModel: Gemma-2-2B\nSAE: {RELEASE} (attention)\n")
        f.write(f"Layer: {layer}\nTask: {task_name}\nd_sae: {num_latents}\nSamples: {len(all_pcts)}\n\n")
        f.write(f"{'Threshold':<12} {'Mean Count':<15} {'Mean %':<10} {'Median %':<10} {'Max %':<10}\n")
        f.write("-" * 60 + "\n")
        for thr, cnt, mp, mdp, mxp in sweep:
            f.write(f"{thr:<12.3f} {cnt:<15.1f} {mp:<10.2f} {mdp:<10.2f} {mxp:<10.2f}\n")
    print(f"Text: {base}.txt")

# ── Run ───────────────────────────────────────────────────────
for task_key, cfg in TASKS.items():
    print(f"\n{'#'*70}")
    print(f"# {cfg['name']} (Layer {cfg['layer']})")
    print(f"{'#'*70}")
    pcts, counts, vals, pooled, n_lat = compute_per_sample(task_key, cfg['layer'], all_sentences[task_key])
    report_and_plot(task_key, cfg['name'], cfg['layer'], pcts, counts, vals, pooled, n_lat)

print("\n\nAll done.")
