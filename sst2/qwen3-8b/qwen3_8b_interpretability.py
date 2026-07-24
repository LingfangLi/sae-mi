# -*- coding: utf-8 -*-
"""Qwen3-8B-Base + Qwen-Scope Residual SAE — Interpretability Score (max-pool)

Old single-file style (mirrors sst2/gemma2-2b/interpretability_score_att.py) but:
  - Uses HF forward hook on residual stream (no TransformerLens dependency)
  - Ranks top-k features by MEAN of per-sample MAX-POOL activation over val set
    (matches paper Eq 2: a_i = max_t z_i^{(t)}, rank features by mean of a_i)
  - Selects top-N exemplars for interpretation by MAX-POOL score (not sum)
  - Held-out Pearson uses MAX-POOL activation vs Llama's 0-10 estimate

Stage-2 script: run AFTER qwen3_8b_pretrained.py has produced probe accuracies,
then set LAYERS_TO_ANALYZE to the 2-3 best layers via --layers CLI.

Model:  Qwen/Qwen3-8B-Base
SAE:    qwen-scope-3-8b-base-w64k-l100  (residual, d_sae=65536, sae_id=layer{N})
"""

import os
import re
import gc
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE
from groq import Groq
from scipy.stats import pearsonr

# ================================================================
# 0. CLI
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=str, required=True,
                    help="Comma-separated ints, e.g. '18,20,22' — best layers from Stage 1")
parser.add_argument("--top_k", type=int, default=10,
                    help="Number of top global features per layer to interpret")
parser.add_argument("--top_n", type=int, default=20,
                    help="Top-N exemplar sentences to keep per feature "
                         "(first 5 → Llama interpretation, next 5 → held-out Pearson)")
parser.add_argument("--val_n", type=int, default=872,
                    help="Number of SST-2 val sentences to use (max 872)")
args = parser.parse_args()

LAYERS_TO_ANALYZE = [int(x) for x in args.layers.split(",")]
TOP_K = args.top_k
TOP_N = args.top_n

# ================================================================
# 1. Paths & env
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ.setdefault("HF_HOME", CACHE_DIR)
os.environ.setdefault("TRANSFORMERS_CACHE", CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", CACHE_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

BASE_MODEL_ID = "Qwen/Qwen3-8B-Base"
SAE_RELEASE = "qwen-scope-3-8b-base-w64k-l100"

# ================================================================
# 2. Groq client
# ================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("Set GROQ_API_KEY environment variable.")

groq_client = Groq(api_key=GROQ_API_KEY)


def chat_with_llama(prompt, model="llama-3.1-8b-instant", max_tokens=150,
                    temperature=0.7, max_retries=5):
    import time
    for attempt in range(max_retries):
        try:
            resp = groq_client.chat.completions.create(
                messages=[
                    {"role": "system",
                     "content": "You are a helpful interpretability assistant."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                wait = 30 * (attempt + 1)
                print(f"  [Groq] rate-limited, sleeping {wait}s ({attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq rate limit exceeded after retries.")


# ================================================================
# 3. Load model + SST-2 val
# ================================================================
print(f"\nLoading {BASE_MODEL_ID} (bf16) ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, torch_dtype=torch.bfloat16,
    device_map=device, cache_dir=CACHE_DIR,
)
model.eval()
n_layers = model.config.num_hidden_layers
d_model = model.config.hidden_size
print(f"Model has {n_layers} layers, d_model={d_model}")

print("\nLoading SST-2 val ...")
val_ds = load_dataset("nyu-mll/glue", "sst2", split="validation", cache_dir=CACHE_DIR)
val_sentences = list(val_ds["sentence"])[: args.val_n]
print(f"Using {len(val_sentences)} val sentences")


@torch.no_grad()
def get_layer_rep(text, layer_idx):
    """Return residual stream at post-block-layer_idx: shape (1, seq, d_model)."""
    enc = tokenizer(text, return_tensors="pt", padding=False,
                    truncation=True, max_length=128).to(device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    return out.hidden_states[layer_idx + 1]


# ================================================================
# 4. SAE-level helpers  (all max-pool over sequence)
# ================================================================
@torch.no_grad()
def rank_topk_by_maxpool(layer_num, sae, k):
    """
    Pass-1 through val set: for each sentence, per-feature max activation
    over tokens; rank features by MEAN of those per-sample maxes.

    Returns: list of top-k feature indices (int).
    """
    d_sae = int(sae.cfg.d_sae)
    running_sum = torch.zeros(d_sae, dtype=torch.float32, device=device)
    for text in tqdm(val_sentences, desc=f"Rank features L{layer_num}"):
        h = get_layer_rep(text, layer_num)                 # (1, seq, d_model)
        z = sae.encode(h.to(sae.dtype))                     # (1, seq, d_sae)
        per_sample_max = z[0].max(dim=0).values.to(torch.float32)  # (d_sae,)
        running_sum += per_sample_max
    mean_score = (running_sum / len(val_sentences)).cpu().numpy()
    topk_idx = np.argsort(mean_score)[::-1][:k]
    print(f"  L{layer_num} top-{k} features (idx, mean_max_act):")
    for idx in topk_idx:
        print(f"    F{int(idx):>6}  {mean_score[idx]:.4f}")
    return [int(i) for i in topk_idx]


@torch.no_grad()
def extract_per_token_acts_for_features(layer_num, sae, feature_ids):
    """
    Pass-2 through val set: for each sentence, cache per-token activations
    ONLY for the requested feature_ids.

    Returns: dict {feature_id: [np.ndarray(seq,), ...]}  (one array per sentence)
             and val_sentences (aligned).
    """
    feat_ids_t = torch.tensor(feature_ids, device=device, dtype=torch.long)
    per_feat_acts = {fid: [] for fid in feature_ids}

    for text in tqdm(val_sentences, desc=f"Extract L{layer_num} top-{len(feature_ids)}"):
        h = get_layer_rep(text, layer_num)                 # (1, seq, d_model)
        z = sae.encode(h.to(sae.dtype))                     # (1, seq, d_sae)
        sub = z[0][:, feat_ids_t].to(torch.float32).cpu().numpy()  # (seq, k)
        for j, fid in enumerate(feature_ids):
            per_feat_acts[fid].append(sub[:, j])
    return per_feat_acts


def maxpool_select_top(acts, texts, n):
    """Select top-N sentences by MAX-POOL activation."""
    scores = np.array([float(np.max(a)) for a in acts])
    order = np.argsort(scores)[::-1][:n]
    return [acts[i] for i in order], [texts[i] for i in order], scores[order]


def llama_interpretation(top_texts, top_acts):
    """Ask Llama to describe what causes strong activation, given per-token acts."""
    prompt = (
        "You are analyzing a sparse autoencoder feature from Qwen3-8B-Base trained on "
        "sentiment data (SST-2).\n"
        "Each sentence below is accompanied by per-token activation strengths for this "
        "feature; larger numbers indicate stronger activation.\n\n"
        "From this data, give ONE concise sentence describing what sentiment pattern or "
        "linguistic concept this feature responds to.\n\n"
    )
    for i, (txt, acts) in enumerate(zip(top_texts, top_acts), 1):
        # cap the numeric list length for prompt size
        acts_short = np.round(acts, 3).tolist()
        prompt += f"{i}. \"{txt}\"\nActivations: {acts_short}\n\n"
    prompt += "Your explanation:"
    return chat_with_llama(prompt)


def llama_activation_score(sentence, interpretation):
    prompt = (
        f'Feature interpretation:\n"{interpretation}"\n\n'
        "On a scale from 0 (not active) to 10 (very active), estimate how strongly "
        "this feature activates on the following sentence. Respond with only a single number.\n\n"
        f'Sentence: "{sentence}"\nActivation:'
    )
    resp = chat_with_llama(prompt, max_tokens=16, temperature=0.0)
    m = re.search(r"\d+(\.\d+)?", resp)
    return float(m.group()) if m else 0.0


def pearson_maxpool(actual_acts, pred_scores):
    """Pearson between per-sample MAX-POOL activation and Llama's 0-10 estimate."""
    actual = np.array([float(np.max(a)) for a in actual_acts])
    pred = np.array(pred_scores)
    if actual.std() == 0 or pred.std() == 0:
        return float("nan")
    corr, _ = pearsonr(actual, pred)
    return float(corr)


def save_report(path, layer_num, feature_idx, interpretation,
                interp_texts, interp_scores,
                eval_texts, eval_acts, pred_scores, corr):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Layer {layer_num}, Feature {feature_idx}\n")
        f.write("-" * 70 + "\n")
        f.write("Interpretation (Llama-3.1-8B):\n" + interpretation.strip() + "\n\n")
        f.write("Top-5 exemplars (used for interpretation):\n")
        for i, (txt, s) in enumerate(zip(interp_texts, interp_scores), 1):
            f.write(f"  {i}. [max={s:.3f}] {txt}\n")
        f.write("\nHeld-out sentences (Pearson evaluation):\n")
        for i, (txt, a, p) in enumerate(zip(eval_texts, eval_acts, pred_scores), 1):
            f.write(f"  {i}. [actual_max={float(np.max(a)):.3f}, pred={p:.3f}] {txt}\n")
        f.write(f"\nPearson correlation (max-pool activation vs Llama 0-10): {corr:.4f}\n")
        f.write("=" * 70 + "\n\n")


def interpretability_for_feature(layer_num, feature_idx,
                                 per_sentence_acts, all_texts, report_path):
    """
    per_sentence_acts: list of np.ndarray (per-token acts for this feature) — one per sentence
    all_texts: list of val sentences (aligned)
    """
    # Top-N by max-pool
    top_acts, top_texts, top_scores = maxpool_select_top(
        per_sentence_acts, all_texts, n=TOP_N
    )
    # First 5 → interpretation
    interp_acts = top_acts[:5]
    interp_texts = top_texts[:5]
    interp_scores = top_scores[:5]

    if float(np.max(interp_scores)) <= 0:
        print(f"  L{layer_num} F{feature_idx}: dead feature, skip")
        return None

    print(f"\n  Top-5 for L{layer_num} F{feature_idx}:")
    for i, (s, sc) in enumerate(zip(interp_texts, interp_scores), 1):
        print(f"    {i}. [max={sc:.3f}] {s}")

    interpretation = llama_interpretation(interp_texts, interp_acts)
    print(f"  → {interpretation}")

    # Held-out: rank 6-10 (max-pool), skipping the exemplars just used
    used = set(interp_texts)
    rest = [(a, t) for a, t in zip(per_sentence_acts, all_texts) if t not in used]
    if not rest:
        print(f"  L{layer_num} F{feature_idx}: no held-out sentences, skip")
        return None
    rest_acts, rest_texts = zip(*rest)
    rest_scores = np.array([float(np.max(a)) for a in rest_acts])
    eval_idx = np.argsort(rest_scores)[::-1][:5]
    eval_acts = [rest_acts[i] for i in eval_idx]
    eval_texts = [rest_texts[i] for i in eval_idx]

    pred_scores = [llama_activation_score(s, interpretation) for s in eval_texts]
    corr = pearson_maxpool(eval_acts, pred_scores)
    print(f"  Pearson (max-pool): {corr:.4f}")

    save_report(report_path, layer_num, feature_idx, interpretation,
                interp_texts, interp_scores.tolist(),
                eval_texts, eval_acts, pred_scores, corr)
    return corr


# ================================================================
# 5. Main loop over selected layers
# ================================================================
overall = {}

for layer in LAYERS_TO_ANALYZE:
    print("\n" + "=" * 70)
    print(f"LAYER {layer}: loading SAE {SAE_RELEASE}/layer{layer}")
    print("=" * 70)

    sae_id = f"layer{layer}"                    # no underscore for qwen-scope
    loaded = SAE.from_pretrained(SAE_RELEASE, sae_id)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    sae.to(device).eval()
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}, k={getattr(sae.cfg,'k','?')}")

    # Pass 1 — rank top-k by max-pool
    topk_feats = rank_topk_by_maxpool(layer, sae, k=TOP_K)

    # Pass 2 — cache per-token acts for only the top-k
    per_feat_acts = extract_per_token_acts_for_features(layer, sae, topk_feats)

    report_file = os.path.join(
        RESULTS_DIR, f"layer_{layer}_top{TOP_K}_interpretability_maxpool.txt"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"QWEN3-8B-BASE INTERPRETABILITY REPORT (max-pool)\n")
        f.write(f"Layer {layer}, top-{TOP_K} global features by mean per-sample max activation\n")
        f.write(f"SAE: {SAE_RELEASE} / {sae_id}\n")
        f.write("=" * 70 + "\n\n")

    correlations = []
    for feat in topk_feats:
        corr = interpretability_for_feature(
            layer, feat, per_feat_acts[feat], val_sentences, report_file
        )
        if corr is not None and not np.isnan(corr):
            correlations.append(corr)

    mean_r = float(np.mean(correlations)) if correlations else float("nan")
    print(f"\nLayer {layer} done. Mean Pearson (max-pool): {mean_r:.4f}")

    with open(report_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"SUMMARY: layer {layer}, features analyzed = {len(correlations)}/{len(topk_feats)}\n")
        f.write(f"Mean Pearson (max-pool): {mean_r:.4f}\n")
        f.write("=" * 70 + "\n")

    overall[layer] = {"mean_pearson": mean_r,
                      "n_valid": len(correlations),
                      "n_total": len(topk_feats)}

    del sae
    gc.collect()
    torch.cuda.empty_cache()

# ================================================================
# 6. Overall summary
# ================================================================
summary_file = os.path.join(RESULTS_DIR, "interpretability_summary_maxpool.txt")
with open(summary_file, "w") as f:
    f.write("=" * 70 + "\n")
    f.write(f"OVERALL SUMMARY — Qwen3-8B interpretability (max-pool)\n")
    f.write(f"SAE: {SAE_RELEASE}\n")
    f.write("=" * 70 + "\n\n")
    for L in sorted(overall):
        f.write(f"  L{L:2d}: mean_r={overall[L]['mean_pearson']:.4f}  "
                f"(valid {overall[L]['n_valid']}/{overall[L]['n_total']})\n")

print(f"\nAll done. Reports in {RESULTS_DIR}")
