import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# === Configuration ===
MODEL_PATH = ""
SAVE_PATH = ""

# === Load tokenizer ===

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# === Load model in original precision (FP32) ===
print("Loading model in FP32 precision...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# === Convert model weights to bfloat16 ===
print("Converting model weights to bfloat16...")
model = model.to(torch.bfloat16)

# === Save converted model and tokenizer ===
print(f"Saving bfloat16 model to {SAVE_PATH}...")
model.save_pretrained(
    SAVE_PATH,
    safe_serialization=True,
    max_shard_size="8GB"
)

tokenizer.save_pretrained(SAVE_PATH)

print("Conversion complete! BF16 model saved successfully.")