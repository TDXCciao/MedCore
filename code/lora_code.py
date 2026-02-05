# -*- coding: utf-8 -*-

import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoConfig,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig


################ Hyperparameter Configuration (Feel free to adjust eval_steps!) ################
model_name      = ""
train_path      = ""
output_dir      = ""
num_epochs      = 3
per_device_batch_size = 32
gradient_accumulation_steps = 4
learning_rate   = 3.0e-5
max_seq_length  = 2048


eval_steps      = 800
save_steps      = 800
early_stopping_patience = 5
validation_ratio = 0.1
################################################################################################


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=True)
tokenizer.pad_token = tokenizer.eos_token


############## 1. 4-bit Quantization ##############
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
    if 'rope_type' in config.rope_scaling:
        config.rope_scaling = {"type": "dynamic", "factor": config.rope_scaling.get("factor", 8.0)}

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    quantization_config=nf4_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model = prepare_model_for_kbit_training(model)


############## 2. LoRA Configuration ##############
lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


############## 3. Dataset Preparation ##############
def format_and_tokenize(example):
    text = f"<|im_start|>user\n{example['input']}<|im_end|>\n<|im_start|>assistant\n{example['output']}<|im_end|>"
    return tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)

raw_dataset = load_dataset("json", data_files=train_path)["train"]
split_dataset = raw_dataset.train_test_split(test_size=validation_ratio, seed=42)
train_data = split_dataset["train"].map(
    format_and_tokenize,
    remove_columns=split_dataset["train"].column_names
)
eval_data = split_dataset["test"].map(
    format_and_tokenize,
    remove_columns=split_dataset["test"].column_names
)

# Optional: Print sample tokenized examples
print("\nValidation: First 3 tokenized training samples (decoded):")
for i in range(min(3, len(train_data))):
    decoded = tokenizer.decode(train_data[i]['input_ids'], skip_special_tokens=False)




############## 4. Training Arguments ##############
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=False,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=per_device_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    save_steps=save_steps,
    learning_rate=learning_rate,
    bf16=True,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    save_total_limit=3,  # Optional: Limits checkpoint storage
)


############## 5. Trainer Setup ##############
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
)


############## 6. Training (Supports Resuming from Checkpoint) ##############
model.print_trainable_parameters()

# Start training — set resume_from_checkpoint=True to continue from latest checkpoint
# Here we explicitly disable resuming to start fresh
trainer.train(resume_from_checkpoint=False)

# Save final model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Training completed. Best checkpoint saved to: {output_dir}")