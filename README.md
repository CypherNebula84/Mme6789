@echo off
REM -----------------------------------
REM Friendly CPU LoRA Trainer for GGUF Models
REM Detects model type, adapts settings, asks user confirmation
REM -----------------------------------

echo Welcome to the CPU LoRA Trainer!
echo.

REM Ask user if they want to auto-detect GGUF model path
set /p AUTO_DETECT="Do you want me to auto-detect GGUF models in this folder? (y/n): "

if /i "%AUTO_DETECT%"=="y" (
    REM Search for .gguf files in current directory
    set MODEL_FILE=
    for %%f in (*.gguf) do (
        set MODEL_FILE=%%f
        goto :found
    )
    :found
    if defined MODEL_FILE (
        echo Found model: %MODEL_FILE%
    ) else (
        echo No GGUF model found. Please enter path manually.
        set /p MODEL_FILE="Enter full path to your GGUF model: "
    )
) else (
    set /p MODEL_FILE="Enter full path to your GGUF model: "
)

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b
)

echo Installing required Python packages (this may take a minute)...
pip install --upgrade pip
pip install torch transformers datasets peft sentencepiece psutil tqdm --quiet

REM Run embedded Python
python - <<END
import os
import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

print("Detecting system resources...")
ram_gb = psutil.virtual_memory().available / 1e9
threads = os.cpu_count()
print(f"Detected ~{ram_gb:.1f} GB RAM and {threads} CPU threads.")

# Load model
model_path = r"%MODEL_FILE%"
print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

try:
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
except:
    print("Forcing CPU float32 load...")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
model.to("cpu")

# Detect dtype
dtype = model.dtype
print(f"Detected model dtype: {dtype}")

# Estimate size
params = sum(p.numel() for p in model.parameters())
model_gb = params * 4 / 1e9
print(f"Estimated FP32 model size: {model_gb:.2f} GB")

# Adaptive settings
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

print(f"Suggested settings: batch_size={batch_size}, LoRA rank={lora_r}, seq_len={seq_len}")
use_defaults = input("Do you want to use these settings? (y/n): ").strip().lower()
if use_defaults != "y":
    batch_size = int(input("Enter batch size: "))
    lora_r = int(input("Enter LoRA rank: "))
    seq_len = int(input("Enter sequence length: "))

# Apply LoRA
print("Applying LoRA...")
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

# Load dataset
print("Loading dataset...")
dataset_name = "wikitext"
dataset = load_dataset(dataset_name, split="train")

def tokenize(batch):
    return tokenizer(batch['text'], truncation=True, max_length=seq_len, padding="max_length")

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Training with progress bar
output_dir = "./lora_cpu_model"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    logging_steps=1,
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
for _ in tqdm(range(training_args.num_train_epochs), desc="Epochs"):
    trainer.train()
print("Saving model...")
trainer.save_model(output_dir)
print("All done! Your LoRA-trained model is in ./lora_cpu_model")
END

pause
