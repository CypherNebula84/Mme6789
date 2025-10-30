[# Mme6789](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
https://huggingface.co/Qwen/Qwen3-4B-GGUF
https://huggingface.co/bartowski/Qwen_Qwen3-4B-Thinking-2507-GGUF
https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/tree/main

https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/



@echo off
REM -----------------------------------
REM Fully Automated CPU LoRA Training for GGUF Models
REM -----------------------------------

REM Ask for model path
set /p MODEL_PATH="Enter path to your GGUF model: "

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b
)

REM Install required packages
echo Installing required Python packages...
pip install --upgrade pip
pip install torch transformers datasets peft sentencepiece psutil --quiet

REM Embedded Python script
python - <<END
import os
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# --- Detect system RAM and CPU threads ---
ram_gb = psutil.virtual_memory().available / 1e9
threads = os.cpu_count()
print(f"Detected ~{ram_gb:.1f} GB available RAM and {threads} CPU threads.")

# --- Load model & tokenizer ---
model_name = r"%MODEL_PATH%"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (CPU)...")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.to("cpu")

# --- Estimate model size ---
params = sum(p.numel() for p in model.parameters())
model_gb = params * 4 / 1e9
print(f"Estimated model size in FP32: {model_gb:.2f} GB")

# --- Suggest settings based on RAM ---
if ram_gb > model_gb + 3:
    batch_size = 2
    lora_r = 8
    seq_len = 512
elif ram_gb > model_gb + 2:
    batch_size = 1
    lora_r = 4
    seq_len = 256
else:
    batch_size = 1
    lora_r = 2
    seq_len = 128

print(f"Suggested: batch_size={batch_size}, LoRA rank={lora_r}, seq_len={seq_len}")

# --- User confirmation ---
use_defaults = input("Use suggested settings? (y/n): ").strip().lower()
if use_defaults != "y":
    batch_size = int(input("Enter batch size: "))
    lora_r = int(input("Enter LoRA rank: "))
    seq_len = int(input("Enter max sequence length: "))

# --- Apply LoRA ---
lora_config = LoraConfig(
    r=lora_r,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- Load dataset ---
dataset_name = "wikitext"
print("Loading dataset...")
dataset = load_dataset(dataset_name, split="train")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=seq_len, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# --- Training ---
output_dir = "./lora_cpu_model"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    fp16=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

print("Starting training...")
trainer.train()
print("Saving model...")
trainer.save_model(output_dir)
print("Training complete!")
END

pause
