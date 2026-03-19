"""
GPT-2 Small SST-2: Generate Groq/Llama-3 descriptions for MI+LR-selected SAE features.
Uses pre-saved activation matrices (no GPU needed).
Saves descriptions as JSON for downstream semantic similarity.
"""

import os
import json
import time
import numpy as np
from datasets import load_dataset
from groq import Groq

# ======================================================================
# Config
# ======================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_mlp")
LAYER = 6  # best layer from probing

TOP_N_SENTENCES = 5  # sentences shown to LLM for interpretation
MODEL_NAME = "GPT-2 Small"
SAE_INFO = "SAELens 32k MLP-out"

# ======================================================================
# Groq client
# ======================================================================
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


# ======================================================================
# Load data
# ======================================================================
print("Loading SST-2 validation sentences...")
val_ds = load_dataset("glue", "sst2", split="validation")
sentences = val_ds["sentence"]
labels = val_ds["label"]
print(f"  {len(sentences)} sentences loaded")

print(f"Loading activation matrix for layer {LAYER}...")
act_path = os.path.join(RESULTS_DIR, f"layer_{LAYER}_activations.npy")
activations = np.load(act_path)  # (872, 32768)
print(f"  Shape: {activations.shape}")

print("Loading feature selection...")
fs_path = os.path.join(RESULTS_DIR, f"layer_{LAYER}_feature_selection.npz")
fs = np.load(fs_path)
filtered_features = fs["filtered_features"]  # array of feature indices
filtered_mi = fs["filtered_mi"]
filtered_lr_weights = fs["filtered_lr_weights"]
filtered_act_rates = fs["filtered_act_rates"]
print(f"  {len(filtered_features)} features to describe")

# ======================================================================
# Load existing progress (resume support)
# ======================================================================
output_path = os.path.join(RESULTS_DIR, f"layer_{LAYER}_descriptions.json")
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        descriptions = json.load(f)
    print(f"  Resuming: {len(descriptions)} descriptions already done")
else:
    descriptions = {}

# ======================================================================
# Generate descriptions
# ======================================================================
print(f"\nGenerating descriptions for {len(filtered_features)} features...")
print("=" * 70)

for i, feat_idx in enumerate(filtered_features):
    feat_key = str(int(feat_idx))

    # Skip if already done
    if feat_key in descriptions:
        continue

    # Get activation column for this feature
    feat_acts = activations[:, feat_idx]  # (872,)
    n_active = (feat_acts > 0).sum()
    act_rate = n_active / len(feat_acts)

    # Find top-N activating sentences
    top_idxs = np.argsort(feat_acts)[::-1][:TOP_N_SENTENCES]
    top_sents = [sentences[j] for j in top_idxs]
    top_vals = [float(feat_acts[j]) for j in top_idxs]
    top_labels = [int(labels[j]) for j in top_idxs]

    # Build prompt
    prompt = (
        f"You are analyzing a sparse autoencoder feature from {MODEL_NAME} ({SAE_INFO}).\n"
        f"This feature (index {feat_idx}, layer {LAYER}) activates on {n_active}/{len(feat_acts)} "
        f"sentences ({act_rate:.1%} activation rate).\n\n"
        f"Below are the top {TOP_N_SENTENCES} sentences where this feature activates most strongly, "
        f"with their activation values and sentiment labels (0=negative, 1=positive).\n\n"
    )
    for j, (sent, val, lab) in enumerate(zip(top_sents, top_vals, top_labels), 1):
        prompt += f'{j}. "{sent}" (activation: {val:.3f}, label: {lab})\n'

    prompt += (
        "\nGive ONE concise sentence describing what linguistic pattern, concept, or "
        "semantic feature this SAE feature responds to. Focus on what the top-activating "
        "sentences have in common."
    )

    # Call Groq
    description = chat_with_llama(prompt)

    # Store result
    descriptions[feat_key] = {
        "feature_index": int(feat_idx),
        "layer": LAYER,
        "description": description,
        "activation_rate": float(act_rate),
        "mi_score": float(filtered_mi[i]),
        "lr_weight": float(filtered_lr_weights[i]),
        "top_sentences": [
            {"sentence": s, "activation": v, "label": l}
            for s, v, l in zip(top_sents, top_vals, top_labels)
        ],
    }

    # Save incrementally
    with open(output_path, "w") as f:
        json.dump(descriptions, f, indent=2, ensure_ascii=False)

    progress = len(descriptions)
    total = len(filtered_features)
    print(f"[{progress}/{total}] Feature {feat_idx}: {description[:80]}...")

    # Small delay to avoid rate limits
    time.sleep(2)

# ======================================================================
# Summary
# ======================================================================
print("\n" + "=" * 70)
print(f"DONE: {len(descriptions)}/{len(filtered_features)} descriptions generated")
print(f"Saved to: {output_path}")

# Also save a human-readable report
report_path = os.path.join(RESULTS_DIR, f"layer_{LAYER}_descriptions_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"SAE Feature Descriptions - {MODEL_NAME}, Layer {LAYER}\n")
    f.write(f"Features: {len(descriptions)} (MI+LR selected, sparsity filtered)\n")
    f.write("=" * 70 + "\n\n")

    for feat_key in sorted(descriptions.keys(), key=lambda x: int(x)):
        d = descriptions[feat_key]
        f.write(f"Feature {d['feature_index']} "
                f"(MI={d['mi_score']:.4f}, LR={d['lr_weight']:.4f}, "
                f"act_rate={d['activation_rate']:.3f})\n")
        f.write(f"  Description: {d['description']}\n")
        f.write(f"  Top sentences:\n")
        for s in d["top_sentences"]:
            f.write(f"    [{s['label']}] ({s['activation']:.3f}) {s['sentence']}\n")
        f.write("\n")

print(f"Report saved to: {report_path}")
