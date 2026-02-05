import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import json
import numpy as np
from tqdm import tqdm
import csv
import os

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = ""  # ←←← Fill this
DEVICE = "cuda"
BATCH_SIZE = 8
TOP_PERCENT_LOW = 0.2
TOP_PERCENT_HIGH = 0.2
MAX_LENGTH = 2048
EMBED_NOISE_STD = 0.03  # ←←← NEW: same as Stage 3

DATASET_PATH = "your_dataset.jsonl"  # ←←← UPDATE THIS

# -----------------------------
# Load Dataset
# -----------------------------
dataset = []
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line.strip()))
print(f"Loaded {len(dataset)} samples from {DATASET_PATH}")

# -----------------------------
# Tokenizer & Model Setup
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, lora_config)
model.eval()  # We'll temporarily switch to train() in compute_grad_norm


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ADD EMBEDDING NOISE HOOK (same as Stage 3)
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def add_embedding_noise(module, input, output):
    # Use global flag or model.training; we'll control via model.train()/eval()
    if model.training and EMBED_NOISE_STD > 0:
        noise = torch.randn_like(output) * EMBED_NOISE_STD
        output = output + noise
    return output

hook_handle = model.get_input_embeddings().register_forward_hook(add_embedding_noise)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# -----------------------------
# Helper: Compute Gradient Norm with Embedding Perturbation
# -----------------------------
def compute_grad_norm(input_ids: torch.Tensor) -> float:
    """
    Compute L2 norm of gradients w.r.t. LoRA parameters under embedding noise.
    Temporarily sets model to train() mode to activate the noise hook.
    """
    original_training = model.training
    model.train()  # Enable noise (and dropout, but acceptable for grad norm)
    model.zero_grad()

    input_ids = input_ids.unsqueeze(0)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()

    total_norm_sq = 0.0
    for name, param in model.named_parameters():
        if "lora" in name and param.grad is not None:
            total_norm_sq += param.grad.detach().square().sum().item()

    model.eval()  # Restore original mode
    return np.sqrt(total_norm_sq)


# -----------------------------
# Main Loop
# -----------------------------
all_grad_norms = []
all_samples_with_grad = []

for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Computing gradient norms"):
    batch_samples = dataset[i:i + BATCH_SIZE]
    tokenized = tokenizer(
        [s["input"] + " " + s["output"] for s in batch_samples],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None
    )

    batch_grad_norms = []
    for enc in tokenized["input_ids"]:
        input_ids = torch.tensor(enc, dtype=torch.long, device=DEVICE)
        grad_norm = compute_grad_norm(input_ids)
        batch_grad_norms.append(grad_norm)

    for sample, grad_norm in zip(batch_samples, batch_grad_norms):
        sample_copy = sample.copy()
        sample_copy["grad_norm"] = grad_norm
        all_samples_with_grad.append(sample_copy)
        all_grad_norms.append(grad_norm)


# -----------------------------
# Save Results
# -----------------------------
CSV_OUTPUT = "gradient_norms_all.csv"
os.makedirs(os.path.dirname(CSV_OUTPUT) or ".", exist_ok=True)
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "grad_norm"])
    for idx, gn in enumerate(all_grad_norms):
        writer.writerow([idx, gn])

grad_norms_np = np.array(all_grad_norms)
sorted_indices = np.argsort(grad_norms_np)
n_total = len(grad_norms_np)
low_cutoff = int(n_total * TOP_PERCENT_LOW)
high_cutoff = int(n_total * (1 - TOP_PERCENT_HIGH))
keep_indices = sorted_indices[low_cutoff:high_cutoff]
filtered_dataset = [all_samples_with_grad[i] for i in keep_indices]

FILTERED_OUTPUT = "gradient_filtered_dataset.jsonl"
with open(FILTERED_OUTPUT, "w", encoding="utf-8") as f:
    for sample in filtered_dataset:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"Original dataset size: {len(dataset)}")
print(f"Filtered dataset size: {len(filtered_dataset)}")
print(f"Filtered dataset saved to: {FILTERED_OUTPUT}")

# Clean up
hook_handle.remove()