# -*- coding: utf-8 -*-
"""Gemma-2B Attention SAE - MRPC Interpretability Score (Original Method)

Stage 2: For selected best layers, find top-10 global features by frequency,
generate Llama3 interpretations via Groq, predict activation scores,
compute Pearson correlation as interpretability score.

- Model: google/gemma-2-2b (TransformerLens)
- SAE: gemma-scope-2b-pt-att-canonical (attention hook)
- Groq/Llama3 for interpretation + activation prediction
"""

# ================================================================
# 0. Imports
# ================================================================
import os
import re
import gc
import time
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader
from sae_lens import SAE
from transformer_lens import HookedTransformer
from groq import Groq
from scipy.stats import pearsonr
from tqdm import tqdm
from huggingface_hub import login

# ================================================================
# 1. Config
# ================================================================
# UPDATE these after Stage 1 results: pick the 2 best-performing layers
LAYERS_TO_ANALYZE = [9, 10]
TOP_K = 10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_att")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ================================================================
# 2. Groq / Llama-3 Client
# ================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("Set GROQ_API_KEY environment variable.")

client = Groq(api_key=GROQ_API_KEY)


def chat_with_llama(prompt, model="llama-3.1-8b-instant", max_tokens=150):
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful interpretability assistant."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/5)...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq API rate limit exceeded after 5 retries")


# ================================================================
# 3. Load Gemma-2B & MRPC Validation
# ================================================================
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    print("Logged in to Hugging Face")

print("Loading HookedTransformer (Gemma-2-2B)...")
model = HookedTransformer.from_pretrained("google/gemma-2-2b", device=device)
model.eval()

print("Loading MRPC validation split...")
val_ds = load_dataset("glue", "mrpc", split="validation")

SEP = "<|endoftext|>"
val_sentences = [f"{ex['sentence1']} {SEP} {ex['sentence2']}" for ex in val_ds]
print(f"Loaded {len(val_sentences)} validation sentence pairs")

# Tokenize with Gemma
tokens_all = model.to_tokens(val_sentences, prepend_bos=True, truncate=True)
print(f"Tokenized shape: {tokens_all.shape}")


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, tokens):
        self.tokens = tokens

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx]


val_tok_ds = TokenDataset(tokens_all)

torch.cuda.empty_cache()
gc.collect()

# ================================================================
# 4. SAE Wrapper
# ================================================================
class SAEWithActs(torch.nn.Module):
    """Wrap an SAE to expose latent activations via hook_sae_acts_post."""

    def __init__(self, sae_model):
        super().__init__()
        self.sae = sae_model
        self.activations = None
        hook_module = getattr(self.sae, "hook_sae_acts_post", None)
        if hook_module is None:
            raise ValueError("hook_sae_acts_post not found on SAE model.")
        hook_module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activations = out.detach()

    def forward(self, sae_input):
        _ = self.sae(sae_input)
        return self.activations

    def get_acts(self, sae_input):
        self.eval()
        with torch.no_grad():
            _ = self.forward(sae_input)
            return self.activations


# ================================================================
# 5. Top-K Global Features by Frequency
# ================================================================
def top_k_global_features(layer_num, sae_model, k=10):
    """Count global feature firing frequency across validation set."""
    sae_wrap = SAEWithActs(sae_model).to(device)
    hook_name = f"blocks.{layer_num}.attn.hook_z"
    counter = Counter()
    dataloader = DataLoader(val_tok_ds, batch_size=8, shuffle=False)

    for batch_tokens in tqdm(dataloader, desc=f"Counting features L{layer_num}"):
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
            hook_acts = cache[hook_name]
            latents = sae_wrap.get_acts(hook_acts)

            active = (latents > 0).any(dim=1)  # (B, d_sae)
            for row in active:
                idxs = row.nonzero(as_tuple=True)[0].tolist()
                counter.update(idxs)

            del cache, hook_acts, latents
            torch.cuda.empty_cache()

    topk = counter.most_common(k)
    print(f"Layer {layer_num} top-{k} global features (idx, count): {topk}")
    del sae_wrap
    return [idx for idx, _ in topk]


# ================================================================
# 6. Interpretability Helpers
# ================================================================
def extract_feature_acts(layer_num, sae_wrap, feature_idx):
    """Extract per-token activations for a specific feature across val set."""
    hook_name = f"blocks.{layer_num}.attn.hook_z"
    acts_all, texts_all = [], []
    dataloader = DataLoader(val_tok_ds, batch_size=8, shuffle=False)

    for i, batch_tokens in enumerate(tqdm(dataloader, desc=f"Extracting L{layer_num} F{feature_idx}")):
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch_tokens, names_filter=[hook_name])
            hook_acts = cache[hook_name]
            latents = sae_wrap.get_acts(hook_acts)

            feat_acts = latents[:, :, feature_idx].cpu().numpy()

            del cache, hook_acts, latents
            torch.cuda.empty_cache()

        start = i * dataloader.batch_size
        batch_texts = val_sentences[start: start + feat_acts.shape[0]]

        for a_sent, txt in zip(feat_acts, batch_texts):
            acts_all.append(a_sent)
            texts_all.append(txt)

    return acts_all, texts_all


def select_top(acts, texts, n=20):
    scores = [np.sum(a) for a in acts]
    idxs = np.argsort(scores)[::-1][:n]
    return [acts[i] for i in idxs], [texts[i] for i in idxs]


def llama_interpretation(top_texts, top_acts):
    prompt = (
        "You are analyzing a sparse autoencoder feature from Gemma-2B on MRPC paraphrase detection data.\n"
        "Each sentence below is accompanied by per-token activation strengths for this feature; "
        "larger numbers indicate stronger activation.\n\n"
        "From this data, give ONE concise sentence describing what pattern or linguistic concept "
        "this feature responds to.\n\n"
    )
    for i, (txt, acts) in enumerate(zip(top_texts, top_acts), 1):
        prompt += f"{i}. \"{txt[:100]}\"\nActivations: {acts.tolist()}\n\n"
    prompt += "Your explanation:"
    return chat_with_llama(prompt)


def llama_activation_score(sentence, interpretation):
    prompt = (
        f'Feature interpretation:\n"{interpretation}"\n\n'
        "On a scale from 0 (not active) to 10 (very active), estimate how strongly "
        "this feature activates on the following sentence. Respond with only a single number.\n\n"
        f'Sentence: \"{sentence[:100]}\"\nActivation:'
    )
    resp = chat_with_llama(prompt, max_tokens=16)
    m = re.search(r"\d+(\.\d+)?", resp)
    return float(m.group()) if m else 0.0


def pearson_score(actual_acts, pred_scores):
    actual = np.array([np.mean(a) for a in actual_acts]) * 10.0
    pred = np.array(pred_scores)
    if len(actual) < 2:
        return float('nan')
    if np.std(actual) == 0 or np.std(pred) == 0:
        return float('nan')
    corr, _ = pearsonr(actual, pred)
    return corr


def save_report(path, layer_num, feature_idx, interpretation,
                eval_texts, eval_acts, pred_scores, corr):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Layer {layer_num}, Feature {feature_idx}\n")
        f.write("-" * 70 + "\n")
        f.write("Interpretation:\n" + interpretation.strip() + "\n\n")
        f.write("Evaluation sentences:\n")
        for i, (txt, acts, pred) in enumerate(zip(eval_texts, eval_acts, pred_scores), 1):
            actual = np.mean(acts) * 10.0
            f.write(f"{i}. {txt[:100]}\n   Actual={actual:.3f}, Pred={pred:.3f}\n")
        f.write(f"\nPearson correlation: {corr:.4f}\n")
        f.write("=" * 70 + "\n\n")


# ================================================================
# 7. Pipeline for One Feature
# ================================================================
def interpretability_for_feature(layer_num, sae_model, feature_idx, report_path):
    sae_wrap = SAEWithActs(sae_model).to(device)

    # 1) Collect activations for full validation set
    acts_all, texts_all = extract_feature_acts(layer_num, sae_wrap, feature_idx)

    # 2) Top-20 sentences; first 5 for interpretation
    top_acts, top_texts = select_top(acts_all, texts_all, n=20)
    interp_acts, interp_texts = top_acts[:5], top_texts[:5]

    print(f"\nTop 5 sentences for L{layer_num} feature {feature_idx}:")
    for i, s in enumerate(interp_texts, 1):
        print(f"{i}. {s[:80]}...")

    interpretation = llama_interpretation(interp_texts, interp_acts)
    print("\nInterpretation:", interpretation)

    # 3) Evaluation on held-out top-5
    used = set(interp_texts)
    rest = [(a, t) for a, t in zip(acts_all, texts_all) if t not in used]
    if not rest:
        print("No remaining sentences for evaluation; skipping feature.")
        return None

    rest_acts, rest_texts = zip(*rest)
    rest_acts, rest_texts = list(rest_acts), list(rest_texts)
    scores = [np.sum(a) for a in rest_acts]
    sorted_idxs = np.argsort(scores)[::-1]

    MAX_RETRIES = 10
    batch_size = 5
    corr = float('nan')

    for attempt in range(MAX_RETRIES):
        start = attempt * batch_size
        end = start + batch_size
        if end > len(sorted_idxs):
            break

        idxs = sorted_idxs[start:end]
        eval_acts = [rest_acts[i] for i in idxs]
        eval_texts = [rest_texts[i] for i in idxs]

        if attempt == 0:
            print("\nEvaluation sentences:")
            for i, s in enumerate(eval_texts, 1):
                print(f"{i}. {s[:80]}...")

        pred_scores = [llama_activation_score(s, interpretation) for s in eval_texts]
        corr = pearson_score(eval_acts, pred_scores)

        if not np.isnan(corr):
            break
        print(f"  Attempt {attempt+1}: NaN (constant scores), retrying with next batch...")

    print(f"Pearson correlation for L{layer_num} feature {feature_idx}: {corr:.4f}")

    save_report(report_path, layer_num, feature_idx, interpretation,
                eval_texts, eval_acts, pred_scores, corr)

    del sae_wrap
    torch.cuda.empty_cache()
    gc.collect()

    return corr


# ================================================================
# 8. Main: Analyze Selected Layers
# ================================================================
release = "gemma-scope-2b-pt-att-canonical"

for layer in LAYERS_TO_ANALYZE:
    print("\n" + "=" * 70)
    print(f"LAYER {layer}: Loading Gemma Scope SAE and finding top-{TOP_K} global features")
    print("=" * 70)

    torch.cuda.empty_cache()
    gc.collect()

    sae_id = f"layer_{layer}/width_16k/canonical"
    try:
        sae_model = SAE.from_pretrained(release, sae_id)
    except:
        sae_model = SAE.from_pretrained(release, sae_id)[0]
    sae_model.to(device).eval()
    print(f"Loaded SAE: {sae_id}")

    # Dynamically find top-K global features
    topk_feats = top_k_global_features(layer, sae_model, k=TOP_K)

    report_file = os.path.join(RESULTS_DIR, f"layer_{layer}_top{TOP_K}_interpretability.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"GEMMA-2B INTERPRETABILITY REPORT - Layer {layer}, MRPC, top-{TOP_K} global features\n")
        f.write("=" * 70 + "\n\n")

    correlations = []
    for feat in topk_feats:
        print(f"\n--- Layer {layer}, feature {feat} ---")
        corr = interpretability_for_feature(
            layer_num=layer,
            sae_model=sae_model,
            feature_idx=feat,
            report_path=report_file,
        )
        if corr is not None and not np.isnan(corr):
            correlations.append((feat, corr))

    mean_corr = np.mean([c for _, c in correlations]) if correlations else 0.0
    best_feat, best_corr = max(correlations, key=lambda x: x[1]) if correlations else (None, 0.0)

    summary_lines = [
        "\n" + "=" * 70,
        f"SUMMARY: Layer {layer}",
        f"Mean Pearson correlation: {mean_corr:.4f}",
        f"Features analyzed: {len(correlations)}/{len(topk_feats)}",
    ]
    if best_feat is not None:
        summary_lines.append(f"Best feature: {best_feat} (correlation: {best_corr:.4f})")
    summary_lines.append("\nAll correlations:")
    for feat, corr in sorted(correlations, key=lambda x: x[1], reverse=True):
        summary_lines.append(f"  Feature {feat}: {corr:.4f}")
    summary_lines.append("=" * 70)

    summary_text = "\n".join(summary_lines) + "\n"

    with open(report_file, "a", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"Report saved to {report_file}")

    del sae_model
    torch.cuda.empty_cache()
    gc.collect()

print("\n" + "=" * 70)
print("ALL LAYERS COMPLETE")
print("=" * 70)
