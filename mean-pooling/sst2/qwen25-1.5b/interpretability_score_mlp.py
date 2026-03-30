# -*- coding: utf-8 -*-
"""DeepSeek-R1-Distill-Qwen-1.5B MLP SAE — Interpretability Score (Original Method)

Stage 2: For selected best layers, find top-10 global features by frequency,
generate Llama3 interpretations via Groq, predict activation scores,
compute Pearson correlation as interpretability score.

- Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (HuggingFace)
- SAE: EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k (eai-sparsify, MLP hook)
- Groq/Llama3 for interpretation + activation prediction
"""

# ================================================================
# 0. Imports
# ================================================================
import os
import re
import gc
import torch
import numpy as np
from collections import Counter
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from groq import Groq
from scipy.stats import pearsonr
from tqdm import tqdm

# ================================================================
# 1. Config
# ================================================================
# UPDATE these after Stage 1 results: pick the 2 best-performing layers
LAYERS_TO_ANALYZE = [15, 14, 13]
TOP_K = 10

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

# ================================================================
# 2. Groq / Llama-3 Client
# ================================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("Set GROQ_API_KEY environment variable.")

client = Groq(api_key=GROQ_API_KEY)


def chat_with_llama(prompt, model_name="llama-3.1-8b-instant", max_tokens=150):
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful interpretability assistant."},
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


# ================================================================
# 3. Load Model & SST-2 Validation
# ================================================================
base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
sae_repo = "EleutherAI/sae-DeepSeek-R1-Distill-Qwen-1.5B-65k"

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

print("Loading SST-2 validation split...")
val_ds = load_dataset("glue", "sst2", split="validation")
val_sentences = list(val_ds["sentence"])
print(f"Loaded {len(val_sentences)} validation sentences")


# ================================================================
# 4. Text Dataset for DataLoader
# ================================================================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


val_text_ds = TextDataset(val_sentences)


# ================================================================
# 5. Helper: Get MLP Output via Forward Hook
# ================================================================
def get_mlp_output(text, layer_idx):
    enc = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128,
    ).to(device)

    captured = {}

    def hook_fn(module, input, output):
        captured['out'] = output

    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        hook.remove()

    return captured['out']  # (1, seq, 1536)


def get_mlp_output_batch(text_batch, layer_idx, max_length=128):
    """Batch version of get_mlp_output."""
    enc = tokenizer(
        list(text_batch),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    captured = {}

    def hook_fn(module, input, output):
        captured['out'] = output

    hook = model.model.layers[layer_idx].mlp.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**enc, use_cache=False)
    finally:
        hook.remove()

    return captured['out'], enc["attention_mask"]  # (B, seq, 1536), (B, seq)


# ================================================================
# 6. SAE Encode Helper (eai-sparsify -> dense)
# ================================================================
def sae_encode_dense(sae, mlp_out, num_latents):
    """Encode MLP output with sparsify SAE, return dense activations."""
    seq_len = mlp_out.shape[1]
    flat = mlp_out.view(-1, mlp_out.shape[-1])  # (B*seq, d_in)

    enc_output = sae.encode(flat)
    # enc_output.top_acts: (B*seq, k=32)
    # enc_output.top_indices: (B*seq, k=32)

    dense = torch.zeros(flat.shape[0], num_latents, device=device, dtype=torch.float32)
    dense.scatter_(1, enc_output.top_indices.long(), enc_output.top_acts.float())

    # Reshape back to (B, seq, num_latents) if needed
    if len(mlp_out.shape) == 3:
        batch_size = mlp_out.shape[0]
        dense = dense.view(batch_size, seq_len, num_latents)

    return dense


# ================================================================
# 7. Top-K Global Features by Frequency
# ================================================================
def top_k_global_features(layer_num, sae, num_latents, k=10):
    """Count global feature firing frequency across validation set."""
    counter = Counter()
    dataloader = DataLoader(val_text_ds, batch_size=32, shuffle=False)

    for batch_texts in tqdm(dataloader, desc=f"Counting features L{layer_num}"):
        mlp_out, attn_mask = get_mlp_output_batch(batch_texts, layer_num)

        with torch.no_grad():
            latents = sae_encode_dense(sae, mlp_out, num_latents)  # (B, seq, num_latents)

            # Mask padded positions
            if attn_mask is not None:
                mask = attn_mask.unsqueeze(-1)  # (B, seq, 1)
                latents = latents * mask

            active = (latents > 0).any(dim=1)  # (B, num_latents)
            for row in active:
                idxs = row.nonzero(as_tuple=True)[0].tolist()
                counter.update(idxs)

            del mlp_out, latents
            torch.cuda.empty_cache()

    topk = counter.most_common(k)
    print(f"Layer {layer_num} top-{k} global features (idx, count): {topk}")
    return [idx for idx, _ in topk]


# ================================================================
# 8. Interpretability Helpers
# ================================================================
def extract_feature_acts(layer_num, sae, num_latents, feature_idx):
    """Extract per-token activations for a specific feature across val set."""
    acts_all, texts_all = [], []
    dataloader = DataLoader(val_text_ds, batch_size=32, shuffle=False)

    for i, batch_texts in enumerate(tqdm(dataloader, desc=f"Extracting L{layer_num} F{feature_idx}")):
        mlp_out, attn_mask = get_mlp_output_batch(batch_texts, layer_num)

        with torch.no_grad():
            latents = sae_encode_dense(sae, mlp_out, num_latents)  # (B, seq, num_latents)

            if attn_mask is not None:
                mask = attn_mask.unsqueeze(-1)
                latents = latents * mask

            feat_acts = latents[:, :, feature_idx].cpu().numpy()  # (B, seq)

            del mlp_out, latents
            torch.cuda.empty_cache()

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
        "You are analyzing a sparse autoencoder feature from DeepSeek-R1-Distill-Qwen-1.5B.\n"
        "Each sentence below is accompanied by per-token activation strengths for this feature; "
        "larger numbers indicate stronger activation.\n\n"
        "From this data, give one concise sentence describing what pattern or concept "
        "this feature responds to.\n\n"
    )
    for i, (txt, acts) in enumerate(zip(top_texts, top_acts), 1):
        prompt += f"{i}. \"{txt}\"\nActivations: {acts.tolist()}\n\n"
    prompt += "Your explanation:"
    return chat_with_llama(prompt)


def llama_activation_score(sentence, interpretation):
    prompt = (
        f'Feature interpretation:\n"{interpretation}"\n\n'
        "On a scale from 0 (not active) to 10 (very active), estimate how strongly "
        "this feature activates on the following sentence. Respond with only a single number.\n\n"
        f'Sentence: \"{sentence}\"\nActivation:'
    )
    resp = chat_with_llama(prompt, max_tokens=16)
    m = re.search(r"\d+(\.\d+)?", resp)
    return float(m.group()) if m else 0.0


def pearson_score(actual_acts, pred_scores):
    actual = np.array([np.mean(a) for a in actual_acts]) * 10.0
    pred = np.array(pred_scores)
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
            f.write(f"{i}. {txt}\n   Actual={actual:.3f}, Pred={pred:.3f}\n")
        f.write(f"\nPearson correlation: {corr:.4f}\n")
        f.write("=" * 70 + "\n\n")


# ================================================================
# 9. Pipeline for One Feature
# ================================================================
def interpretability_for_feature(layer_num, sae, num_latents, feature_idx, report_path):
    # 1) Collect activations for full validation set
    acts_all, texts_all = extract_feature_acts(layer_num, sae, num_latents, feature_idx)

    # 2) Top-20 sentences; first 5 for interpretation
    top_acts, top_texts = select_top(acts_all, texts_all, n=20)
    interp_acts, interp_texts = top_acts[:5], top_texts[:5]

    print(f"\nTop 5 sentences for L{layer_num} feature {feature_idx}:")
    for i, s in enumerate(interp_texts, 1):
        print(f"{i}. {s}")

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
    idxs = np.argsort(scores)[::-1][:5]
    eval_acts = [rest_acts[i] for i in idxs]
    eval_texts = [rest_texts[i] for i in idxs]

    print("\nEvaluation sentences:")
    for i, s in enumerate(eval_texts, 1):
        print(f"{i}. {s}")

    pred_scores = [llama_activation_score(s, interpretation) for s in eval_texts]
    corr = pearson_score(eval_acts, pred_scores)
    print(f"Pearson correlation for L{layer_num} feature {feature_idx}: {corr:.4f}")

    save_report(report_path, layer_num, feature_idx, interpretation,
                eval_texts, eval_acts, pred_scores, corr)

    return corr


# ================================================================
# 10. Main: Analyze Selected Layers
# ================================================================
for layer in LAYERS_TO_ANALYZE:
    print("\n" + "=" * 70)
    print(f"LAYER {layer}: Loading SAE and finding top-{TOP_K} global features")
    print("=" * 70)

    torch.cuda.empty_cache()
    gc.collect()

    hookpoint = f"layers.{layer}.mlp"
    try:
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
        sae.to(device).eval()
        num_latents = sae.num_latents
        print(f"Loaded SAE: {hookpoint} (d_in={sae.d_in}, num_latents={num_latents})")
    except Exception as e:
        print(f"Could not load SAE for layer {layer}: {e}")
        continue

    # Dynamically find top-K global features
    topk_feats = top_k_global_features(layer, sae, num_latents, k=TOP_K)

    report_file = os.path.join(RESULTS_DIR, f"layer_{layer}_top{TOP_K}_interpretability.txt")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"DEEPSEEK-R1-DISTILL-QWEN-1.5B INTERPRETABILITY REPORT - Layer {layer}, "
                f"top-{TOP_K} global features (MLP SAE)\n")
        f.write("=" * 70 + "\n\n")

    correlations = []
    for feat in topk_feats:
        print(f"\n--- Layer {layer}, feature {feat} ---")
        corr = interpretability_for_feature(
            layer_num=layer,
            sae=sae,
            num_latents=num_latents,
            feature_idx=feat,
            report_path=report_file,
        )
        if corr is not None:
            correlations.append(corr)

    mean_corr = np.mean(correlations) if correlations else 0.0
    print(f"\nLayer {layer} finished. Mean Pearson: {mean_corr:.4f}")
    print(f"Report saved to {report_file}")

    with open(report_file, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"SUMMARY: Layer {layer}\n")
        f.write(f"Mean Pearson correlation: {mean_corr:.4f}\n")
        f.write(f"Features analyzed: {len(correlations)}/{len(topk_feats)}\n")
        f.write("=" * 70 + "\n")

    del sae
    torch.cuda.empty_cache()
    gc.collect()

print("\n" + "=" * 70)
print("ALL LAYERS COMPLETE")
print("=" * 70)
