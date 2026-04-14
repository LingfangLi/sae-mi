# -*- coding: utf-8 -*-
"""Plot SAE activation value distributions: TopK (DeepSeek) vs ReLU (Gemma)

Demonstrates why DeepSeek has ~87% feature coverage while Gemma has ~61%.
Root cause: TopK encoding forces exactly K=32 features per token,
while ReLU encoding produces naturally sparse activations.
"""

import os
import gc
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Europarl data (shared) ─────────────────────────────────────
DATA_DIR = "/mnt/scratch/users/yangwr/Lingfang/saes-mi/data"
EN_PATH = os.path.join(DATA_DIR, "europarl-v7.fr-en.en")
FR_PATH = os.path.join(DATA_DIR, "europarl-v7.fr-en.fr")

print("Loading Europarl sentences...")
with open(EN_PATH, 'r') as f:
    en_lines = [l.strip() for l in f if l.strip()][:200]
with open(FR_PATH, 'r') as f:
    fr_lines = [l.strip() for l in f if l.strip()][:200]

N_SAMPLES = 100
sentences_en_fr = [f"{en} {fr}" for en, fr in zip(en_lines[:N_SAMPLES], fr_lines[:N_SAMPLES])]
print(f"Using {N_SAMPLES} Europarl sentence pairs")

# ================================================================
# 1. DeepSeek-R1-Distill-Qwen-1.5B + sparsify TopK SAE
# ================================================================
print("\n" + "="*70)
print("1. DeepSeek (sparsify TopK=32 SAE, 65k features)")
print("="*70)

from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae

ds_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ds_tokenizer = AutoTokenizer.from_pretrained(ds_model_id, cache_dir=CACHE_DIR)
if ds_tokenizer.pad_token is None:
    ds_tokenizer.pad_token = ds_tokenizer.eos_token

ds_model = AutoModelForCausalLM.from_pretrained(
    ds_model_id, torch_dtype=torch.bfloat16, device_map=device, cache_dir=CACHE_DIR
)
ds_model.eval()

LAYER_DS = 14
sae_ds = Sae.load_from_hub(
    "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k",
    hookpoint=f"layers.{LAYER_DS}.mlp"
)
sae_ds.to(device).eval()
num_latents_ds = sae_ds.num_latents
print(f"SAE: d_in={sae_ds.d_in}, num_latents={num_latents_ds}, k={sae_ds.cfg.k}")

# Collect per-token activations
ds_all_acts = []        # all activation values (including zeros) per token
ds_nonzero_counts = []  # number of non-zero features per token
ds_nonzero_vals = []    # non-zero activation values

captured = {}
def hook_fn(module, inp, out):
    captured['out'] = out

hook = ds_model.model.layers[LAYER_DS].mlp.register_forward_hook(hook_fn)

with torch.no_grad():
    for i, sent in enumerate(sentences_en_fr):
        enc = ds_tokenizer(sent, return_tensors="pt", truncation=True, max_length=128).to(device)
        ds_model(**enc, use_cache=False)
        mlp_out = captured['out']  # (1, seq, d)
        flat = mlp_out.view(-1, mlp_out.shape[-1])
        enc_output = sae_ds.encode(flat)

        seq_len = flat.shape[0]
        dense = torch.zeros(seq_len, num_latents_ds, device=device, dtype=torch.float32)
        dense.scatter_(1, enc_output.top_indices.long(), enc_output.top_acts.float())

        for t in range(seq_len):
            token_acts = dense[t].cpu().numpy()
            nz_mask = token_acts > 0
            ds_nonzero_counts.append(nz_mask.sum())
            ds_nonzero_vals.extend(token_acts[nz_mask].tolist())

        if (i+1) % 20 == 0:
            print(f"  DeepSeek: {i+1}/{N_SAMPLES}")

hook.remove()
del ds_model, sae_ds, captured
gc.collect()
torch.cuda.empty_cache()

ds_nonzero_vals = np.array(ds_nonzero_vals)
ds_nonzero_counts = np.array(ds_nonzero_counts)
print(f"DeepSeek: {len(ds_nonzero_counts)} tokens, "
      f"avg active features/token: {ds_nonzero_counts.mean():.1f}, "
      f"median non-zero val: {np.median(ds_nonzero_vals):.4f}")

# ================================================================
# 2. Gemma-2-2B + SAELens ReLU attention SAE
# ================================================================
print("\n" + "="*70)
print("2. Gemma-2-2B (SAELens ReLU SAE, 16k features)")
print("="*70)

from transformer_lens import HookedTransformer
from sae_lens import SAE
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

gemma_model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
gemma_model.eval()

LAYER_GEMMA = 15
hook_name = f"blocks.{LAYER_GEMMA}.attn.hook_z"

try:
    sae_gemma = SAE.from_pretrained("gemma-scope-2b-pt-att-canonical",
                                     f"layer_{LAYER_GEMMA}/width_16k/canonical")
except:
    sae_gemma = SAE.from_pretrained("gemma-scope-2b-pt-att-canonical",
                                     f"layer_{LAYER_GEMMA}/width_16k/canonical")[0]
sae_gemma.to(device).eval()
print(f"SAE: d_sae={sae_gemma.cfg.d_sae}")

gemma_nonzero_counts = []
gemma_nonzero_vals = []

with torch.no_grad():
    for i, sent in enumerate(sentences_en_fr):
        tokens = gemma_model.to_tokens(sent, prepend_bos=True)
        _, cache = gemma_model.run_with_cache(tokens, names_filter=[hook_name])
        hook_acts = cache[hook_name]  # (1, seq, n_heads, d_head)
        sae_feats = sae_gemma.encode(hook_acts)  # (1, seq, d_sae)

        seq_len = sae_feats.shape[1]
        for t in range(seq_len):
            token_acts = sae_feats[0, t].cpu().numpy()
            nz_mask = token_acts > 0
            gemma_nonzero_counts.append(nz_mask.sum())
            gemma_nonzero_vals.extend(token_acts[nz_mask].tolist())

        if (i+1) % 20 == 0:
            print(f"  Gemma: {i+1}/{N_SAMPLES}")

del gemma_model, sae_gemma, cache
gc.collect()
torch.cuda.empty_cache()

gemma_nonzero_vals = np.array(gemma_nonzero_vals)
gemma_nonzero_counts = np.array(gemma_nonzero_counts)
print(f"Gemma: {len(gemma_nonzero_counts)} tokens, "
      f"avg active features/token: {gemma_nonzero_counts.mean():.1f}, "
      f"median non-zero val: {np.median(gemma_nonzero_vals):.4f}")

# ================================================================
# 3. Plot
# ================================================================
print("\n" + "="*70)
print("Generating plots...")
print("="*70)

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

# ── Plot A: Number of active features per token ───────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(ds_nonzero_counts, bins=50, alpha=0.7, color='#e74c3c', label='DeepSeek (TopK=32)', edgecolor='black', linewidth=0.5)
ax1.axvline(ds_nonzero_counts.mean(), color='#c0392b', linestyle='--', linewidth=2,
            label=f'Mean={ds_nonzero_counts.mean():.1f}')
ax1.set_xlabel('Active features per token', fontsize=12)
ax1.set_ylabel('Count (tokens)', fontsize=12)
ax1.set_title('(A) DeepSeek: Active Features per Token\n(sparsify TopK=32, 65k features)', fontsize=11)
ax1.legend(fontsize=10)

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(gemma_nonzero_counts, bins=50, alpha=0.7, color='#3498db', label='Gemma (ReLU)', edgecolor='black', linewidth=0.5)
ax2.axvline(gemma_nonzero_counts.mean(), color='#2980b9', linestyle='--', linewidth=2,
            label=f'Mean={gemma_nonzero_counts.mean():.1f}')
ax2.set_xlabel('Active features per token', fontsize=12)
ax2.set_ylabel('Count (tokens)', fontsize=12)
ax2.set_title('(B) Gemma-2B: Active Features per Token\n(SAELens ReLU, 16k features)', fontsize=11)
ax2.legend(fontsize=10)

# ── Plot B: Non-zero activation value distributions ───────────
ax3 = fig.add_subplot(gs[1, 0])
# Clip for better visualization
ds_clipped = ds_nonzero_vals[ds_nonzero_vals < np.percentile(ds_nonzero_vals, 99)]
ax3.hist(ds_clipped, bins=100, alpha=0.7, color='#e74c3c', edgecolor='black', linewidth=0.3)
ax3.axvline(np.median(ds_nonzero_vals), color='#c0392b', linestyle='--', linewidth=2,
            label=f'Median={np.median(ds_nonzero_vals):.4f}')
ax3.set_xlabel('Activation value (non-zero only)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('(C) DeepSeek: Non-zero Activation Distribution\n(many small values near 0 from TopK tail)', fontsize=11)
ax3.legend(fontsize=10)

ax4 = fig.add_subplot(gs[1, 1])
gemma_clipped = gemma_nonzero_vals[gemma_nonzero_vals < np.percentile(gemma_nonzero_vals, 99)]
ax4.hist(gemma_clipped, bins=100, alpha=0.7, color='#3498db', edgecolor='black', linewidth=0.3)
ax4.axvline(np.median(gemma_nonzero_vals), color='#2980b9', linestyle='--', linewidth=2,
            label=f'Median={np.median(gemma_nonzero_vals):.4f}')
ax4.set_xlabel('Activation value (non-zero only)', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('(D) Gemma-2B: Non-zero Activation Distribution\n(ReLU: naturally sparse, higher values)', fontsize=11)
ax4.legend(fontsize=10)

# ── Summary text ──────────────────────────────────────────────
summary = (
    f"DeepSeek (sparsify TopK=32):  {num_latents_ds} features, "
    f"always 32 active/token, "
    f"median val={np.median(ds_nonzero_vals):.4f}\n"
    f"Gemma-2B (SAELens ReLU):      16384 features, "
    f"avg {gemma_nonzero_counts.mean():.0f} active/token, "
    f"median val={np.median(gemma_nonzero_vals):.4f}\n"
    f"TopK forces activation → high dataset coverage (87%); "
    f"ReLU is naturally sparse → lower coverage (61%)"
)
fig.text(0.5, 0.01, summary, ha='center', fontsize=10, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(os.path.join(SCRIPT_DIR, "activation_distribution_comparison.png"),
            dpi=150, bbox_inches='tight')
plt.savefig(os.path.join(SCRIPT_DIR, "activation_distribution_comparison.pdf"),
            bbox_inches='tight')
print(f"Saved to {SCRIPT_DIR}/activation_distribution_comparison.png")
print(f"Saved to {SCRIPT_DIR}/activation_distribution_comparison.pdf")
print("Done.")
