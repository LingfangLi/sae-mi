"""
Qwen2.5 1.5B SST-2: Extract SAE activation matrices (Max Pooling)
Only runs SAE extraction on specified best layers, saves .npy for downstream MI/Dual-Alignment.
Skips baseline and probing (already done).
"""

import os
import gc
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sparsify import Sae
from tqdm import tqdm
import numpy as np

# ======================================================================
# Config
# ======================================================================
LAYERS_TO_EXTRACT = [13, 14, 15]  # best layers from previous probing results

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

# ======================================================================
# Load model and data
# ======================================================================
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

print("Loading SST-2 dataset...")
dataset = load_dataset("glue", "sst2")
val_dataset = dataset["validation"]  # 872 samples
print(f"Validation samples: {len(val_dataset)}")

# ======================================================================
# Helper: get MLP output via forward hook
# ======================================================================
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

# ======================================================================
# Extract SAE activations with Max Pooling
# ======================================================================
for layer_num in LAYERS_TO_EXTRACT:
    print(f"\n{'='*70}")
    print(f"LAYER {layer_num} - SAE Activation Extraction (Max Pooling)")
    print(f"{'='*70}")

    hookpoint = f"layers.{layer_num}.mlp"
    try:
        sae = Sae.load_from_hub(sae_repo, hookpoint=hookpoint)
        sae.to(device)
        sae.eval()
        num_latents = sae.num_latents
        print(f"Loaded SAE: {hookpoint} (d_in={sae.d_in}, num_latents={num_latents})")
    except Exception as e:
        print(f"Could not load SAE for layer {layer_num}: {e}")
        continue

    all_sae_activations = []
    all_labels = []

    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc=f"SAE L{layer_num}"):
            sentence = val_dataset[idx]["sentence"]
            label = val_dataset[idx]["label"]

            mlp_out = get_mlp_output(sentence, layer_num)  # (1, seq, 1536)
            seq_len = mlp_out.shape[1]

            flat = mlp_out.view(-1, mlp_out.shape[-1])  # (seq, 1536)

            enc_output = sae.encode(flat)
            # enc_output.top_acts: (seq, k=32)
            # enc_output.top_indices: (seq, k=32)

            # Convert sparse to dense: (seq, num_latents)
            dense = torch.zeros(seq_len, num_latents, device=device, dtype=torch.float32)
            dense.scatter_(1, enc_output.top_indices.long(), enc_output.top_acts.float())

            # MAX pooling over sequence dimension
            pooled_features = dense.max(dim=0).values.cpu().numpy()

            all_sae_activations.append(pooled_features)
            all_labels.append(label)

    # Save activation matrix and labels
    activation_matrix = np.stack(all_sae_activations)  # (872, 65536)
    labels_array = np.array(all_labels)

    np.save(os.path.join(RESULTS_DIR, f"layer_{layer_num}_activations.npy"), activation_matrix)
    np.save(os.path.join(RESULTS_DIR, f"layer_{layer_num}_labels.npy"), labels_array)

    print(f"Saved: {activation_matrix.shape} activations, {labels_array.shape} labels")

    # Summary stats
    n_active_per_sample = (activation_matrix > 0).sum(axis=1)
    print(f"Active features per sample: mean={n_active_per_sample.mean():.1f}, "
          f"min={n_active_per_sample.min()}, max={n_active_per_sample.max()}")

    # Free SAE memory
    del sae, all_sae_activations, activation_matrix
    gc.collect()
    torch.cuda.empty_cache()

print(f"\nAll activation matrices saved to {RESULTS_DIR}")
