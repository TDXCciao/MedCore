import os
import json
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from tqdm import tqdm

# =========================
# Configuration
# =========================
MODEL_NAME = ""  # Replace with your base model path
DATA_PATH = ""
OUTPUT_DIR = ""

EPOCHS = 6
BATCH_SIZE = 6
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 1e-4
MAX_SEQ_LENGTH = 400

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EMBEDDING_NOISE_STD = 0.03  # Stddev of Gaussian noise added to input embeddings during training

# >>> Limit dataset size for rapid debugging <<<
MAX_SAMPLES_FOR_DEBUG = None  # Set to e.g., 500 to enable debug mode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)



class QADataset(Dataset):
    def __init__(self, data_path, max_samples=None):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if max_samples is not None and idx >= max_samples:
                    break
                obj = json.loads(line)
                obj["_sample_id"] = idx  # Preserve original index for tracking
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =========================
# Tokenizer & Base Model Setup
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Freeze all base model parameters
for param in model.parameters():
    param.requires_grad = False

# =========================
# Inject LoRA Adapter
# =========================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


# =========================
# Embedding Perturbation Hook
# =========================
def add_embedding_noise(module, input, output):
    """Add Gaussian noise to input embeddings during training."""
    if model.training and EMBEDDING_NOISE_STD > 0:
        noise = torch.randn_like(output) * EMBEDDING_NOISE_STD
        output = output + noise
    return output


# Register forward hook on the embedding layer
embedding_hook_handle = model.get_input_embeddings().register_forward_hook(add_embedding_noise)

# =========================
# Data Loading
# =========================
dataset = QADataset(DATA_PATH, max_samples=MAX_SAMPLES_FOR_DEBUG)


def collate_fn(samples):
    """Collate function to prepare input-output pairs with proper labels."""
    texts = [s["input"] + tokenizer.eos_token + s["output"] for s in samples]
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        max_length=MAX_SEQ_LENGTH
    )
    labels = encodings["input_ids"].clone()
    # Mask padding tokens in loss computation
    labels[encodings["attention_mask"] == 0] = -100
    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
        "original_samples": samples
    }


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# =========================
# Training Loop with Per-Sample Loss Tracking
# =========================
model.train()
all_sample_losses = defaultdict(dict)  # {sample_id: {epoch: loss}}

print(f"Debug mode active: using {len(dataset)} samples (max={MAX_SAMPLES_FOR_DEBUG})")

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch} =====")
    epoch_output_path = os.path.join(OUTPUT_DIR, f"losses_epoch{epoch}.jsonl")
    with open(epoch_output_path, "w", encoding="utf-8") as epoch_file:

        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            original_samples = batch["original_samples"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = (shift_labels != -100).float()

            # Compute per-token loss (ignoring padded positions)
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100
            ).view(shift_labels.shape)

            loss_per_token = loss_per_token * shift_mask
            loss_per_sample = loss_per_token.sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)

            # Backpropagate mean loss (with gradient accumulation)
            (loss_per_sample.mean() / GRADIENT_ACCUMULATION_STEPS).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Log per-sample loss
            for idx, sample in enumerate(original_samples):
                sample_id = sample["_sample_id"]
                loss_val = loss_per_sample[idx].item()
                all_sample_losses[sample_id][epoch] = loss_val

                record = dict(sample)
                record["epoch"] = epoch
                record["loss"] = loss_val
                epoch_file.write(json.dumps(record, ensure_ascii=False) + "\n")

# =========================
# Final Analysis: Detect Forget Events
# =========================
final_output_path = os.path.join(OUTPUT_DIR, "stage3_allfields.jsonl")
with open(final_output_path, "w", encoding="utf-8") as final_file:
    for sample in dataset:
        sample_id = sample["_sample_id"]
        losses = all_sample_losses.get(sample_id, {})

        # Skip if missing initial or final epoch loss
        if not losses or 0 not in losses or (EPOCHS - 1) not in losses:
            continue

        initial_loss = losses[0]
        final_loss = losses[EPOCHS - 1]
        # A "forget event" occurs if loss increases over training
        forget_event = int(final_loss > initial_loss)

        record = dict(sample)
        record["losses"] = losses
        record["forget_event"] = forget_event
        final_file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Stage III debug run completed.")

# =========================
# Save Trained LoRA Adapter
# =========================
lora_adapter_path = os.path.join(OUTPUT_DIR, "lora_adapter")
model.save_pretrained(lora_adapter_path)
tokenizer.save_pretrained(lora_adapter_path)
print(f"LoRA adapter saved to: {lora_adapter_path}")

# Clean up hook
embedding_hook_handle.remove()