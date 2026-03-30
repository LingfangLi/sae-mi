# -*- coding: utf-8 -*-
"""DeepSeek MRPC - Interpretability Score Only (Layers 15 & 16)

Standalone Stage 2 script: skips probing, directly runs Groq interpretability
for the top 2 layers from previous MRPC results.
Includes: rate limit retry, NaN retry, per-layer summary.
"""

import os, re, gc, time
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from tqdm import tqdm
from scipy.stats import pearsonr
from groq import Groq
from torch.utils.data import DataLoader

# ================================================================
# 1. Setup
# ================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_mlp")
CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required.")

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
# 2. Load Model and MRPC
# ================================================================
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
print(f"Loading model: {base_model_id} ...")

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
model.eval()

n_layers = model.config.num_hidden_layers
print(f"Model has {n_layers} layers (0-{n_layers-1})")

print("Loading MRPC dataset...")
dataset = load_dataset("glue", "mrpc")
val_dataset = dataset["validation"]
print(f"Validation samples: {len(val_dataset)}")

def combine_sentences(example):
    return f"{example['sentence1']} {tokenizer.eos_token} {example['sentence2']}"

val_combined = [combine_sentences(ex) for ex in val_dataset]

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        return self.sentences[idx]

val_text_ds = TextDataset(val_combined)

# ================================================================
# 3. Helpers
# ================================================================
def get_mlp_output_batch(text_batch, layer_idx, max_length=128):
    enc = tokenizer(
        list(text_batch), return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(device)
    captured = {}
    def hook_fn(module, input, output):
        captured['out'] = output
    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**enc, output_hidden_states=False, use_cache=False)
    finally:
        hook.remove()
    return captured['out'], enc["attention_mask"]

def top_k_global_features(layer_num, sae_model, k=10):
    counter = Counter()
    dataloader = DataLoader(val_text_ds, batch_size=32, shuffle=False)
    num_latents = sae_model.num_latents

    for batch_texts in tqdm(dataloader, desc=f"Counting features L{layer_num}"):
        mlp_out, attn_mask = get_mlp_output_batch(batch_texts, layer_num)
        B, S, D = mlp_out.shape
        flat = mlp_out.view(-1, D)
        enc_out = sae_model.encode(flat)
        dense = torch.zeros(B * S, num_latents, device=device, dtype=torch.float32)
        dense.scatter_(1, enc_out.top_indices.long(), enc_out.top_acts.float())
        dense = dense.view(B, S, num_latents)

        if attn_mask is not None:
            dense = dense * attn_mask.unsqueeze(-1)

        active = (dense > 0).any(dim=1)
        for row in active:
            idxs = row.nonzero(as_tuple=True)[0].tolist()
            counter.update(idxs)

    topk = counter.most_common(k)
    print(f"Layer {layer_num} top-{k} global features: {topk}")
    return [idx for idx, _ in topk]

def extract_feature_acts(layer_num, sae_model, feature_idx):
    acts_all, texts_all = [], []
    dataloader = DataLoader(val_text_ds, batch_size=32, shuffle=False)
    num_latents = sae_model.num_latents

    for batch_texts in tqdm(dataloader, desc=f"Extract L{layer_num} F{feature_idx}"):
        mlp_out, attn_mask = get_mlp_output_batch(batch_texts, layer_num)
        B, S, D = mlp_out.shape
        flat = mlp_out.view(-1, D)
        enc_out = sae_model.encode(flat)
        dense = torch.zeros(B * S, num_latents, device=device, dtype=torch.float32)
        dense.scatter_(1, enc_out.top_indices.long(), enc_out.top_acts.float())
        dense = dense.view(B, S, num_latents)

        if attn_mask is not None:
            dense = dense * attn_mask.unsqueeze(-1)

        feat_acts = dense[:, :, feature_idx].detach().cpu().numpy()
        for a_sent, txt in zip(feat_acts, batch_texts):
            acts_all.append(a_sent)
            texts_all.append(txt)

    return acts_all, texts_all

# ================================================================
# 4. Interpretation helpers
# ================================================================
def select_top(acts, texts, n=20):
    scores = [np.sum(a) for a in acts]
    idxs = np.argsort(scores)[::-1][:n]
    return [acts[i] for i in idxs], [texts[i] for i in idxs]

def llama_interpretation(top_texts, top_acts):
    prompt = (
        "You are analyzing a sparse autoencoder feature from DeepSeek-R1-Distill-Qwen-1.5B on MRPC paraphrase detection.\n"
        "Each sentence below is accompanied by per-token activation strengths; "
        "larger numbers indicate stronger activation.\n\n"
        "From this data, give one concise sentence describing what pattern or concept "
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
        f.write("-"*70 + "\n")
        f.write("Interpretation:\n" + interpretation.strip() + "\n\n")
        f.write("Evaluation sentences:\n")
        for i, (txt, acts, pred) in enumerate(zip(eval_texts, eval_acts, pred_scores), 1):
            actual = np.mean(acts) * 10.0
            f.write(f"{i}. {txt[:100]}\n   Actual={actual:.3f}, Pred={pred:.3f}\n")
        f.write(f"\nPearson correlation: {corr:.4f}\n")
        f.write("="*70 + "\n\n")

# ================================================================
# 5. Pipeline for One Feature (with NaN retry)
# ================================================================
def interpretability_for_feature(layer_num, sae_model, feature_idx, report_path):
    acts_all, texts_all = extract_feature_acts(layer_num, sae_model, feature_idx)

    top_acts, top_texts = select_top(acts_all, texts_all, n=20)
    interp_acts, interp_texts = top_acts[:5], top_texts[:5]

    print(f"\nTop 5 sentences for L{layer_num} feature {feature_idx}:")
    for i, s in enumerate(interp_texts, 1):
        print(f"{i}. {s[:80]}...")

    interpretation = llama_interpretation(interp_texts, interp_acts)
    print(f"\nInterpretation: {interpretation}")

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
    return corr

# ================================================================
# 6. Main: Analyze Layers 15 & 16
# ================================================================
sae_repo = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"
LAYERS_TO_ANALYZE = [15, 16]
TOP_K = 10

for layer in LAYERS_TO_ANALYZE:
    print("\n" + "="*70)
    print(f"LAYER {layer}: loading SAE and finding top-{TOP_K} global features")
    print("="*70)

    hookpoint = f"layers.{layer}.mlp"
    sae_model = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
    sae_model.to(device).eval()
    print(f"Loaded SAE: {hookpoint} (d_in={sae_model.d_in}, num_latents={sae_model.num_latents})")

    topk_feats = top_k_global_features(layer, sae_model, k=TOP_K)

    report_file = os.path.join(RESULTS_DIR,
        f"deepseek_layer{layer}_top{TOP_K}_global_features_interpretability.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"INTERPRETABILITY REPORT - DeepSeek-R1-Distill-Qwen-1.5B, Layer {layer}, MRPC, top-{TOP_K} global features\n")
        f.write("="*70 + "\n\n")

    correlations = []
    for feat in topk_feats:
        print(f"\n--- Layer {layer}, global feature {feat} ---")
        corr = interpretability_for_feature(layer, sae_model, feat, report_file)
        if corr is not None and not np.isnan(corr):
            correlations.append((feat, corr))

    # Per-layer summary
    mean_corr = np.mean([c for _, c in correlations]) if correlations else 0.0
    best_feat, best_corr = max(correlations, key=lambda x: x[1]) if correlations else (None, 0.0)

    summary_lines = [
        "\n" + "="*70,
        f"SUMMARY: Layer {layer}",
        f"Mean Pearson correlation: {mean_corr:.4f}",
        f"Features analyzed: {len(correlations)}/{len(topk_feats)}",
    ]
    if best_feat is not None:
        summary_lines.append(f"Best feature: {best_feat} (correlation: {best_corr:.4f})")
    summary_lines.append("\nAll correlations:")
    for feat, corr in sorted(correlations, key=lambda x: x[1], reverse=True):
        summary_lines.append(f"  Feature {feat}: {corr:.4f}")
    summary_lines.append("="*70)

    summary_text = "\n".join(summary_lines) + "\n"

    with open(report_file, "a", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"Layer {layer}: finished. Report saved to {report_file}")

    del sae_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nCOMPLETE - DeepSeek MRPC Interpretability Analysis")
