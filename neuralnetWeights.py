import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# --- 1. LOCAL FILE PATHS ---
# Update this if your train.csv is somewhere else
TRAIN_PATH = "train.csv"
OUTPUT_MODEL_DIR = "./akkadian_saved_model"

# --- 2. LOAD & PREPARE DATA ---
print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
train_dataset = Dataset.from_pandas(train_df)

# --- 3. LOAD BASE MODEL ---
print("Downloading base ByT5 model from Hugging Face...")
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- 4. PREPROCESSING ---
def preprocess_function(examples):
    # ByT5 is character-level, so we need a larger max_length (e.g., 512 characters)
    model_inputs = tokenizer(examples["transliteration"], max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["translation"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing data...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)

# --- 5. TRAIN ---
print("Configuring training parameters...")
args = Seq2SeqTrainingArguments(
    output_dir="./training_checkpoints",
    report_to="none",               # Disables Weights & Biases tracking
    eval_strategy="no",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_strategy="no",             # We only care about saving the final model
    fp16=torch.cuda.is_available()  # Uses GPU acceleration if you have an Nvidia card
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_train,
    processing_class=tokenizer,
)

print("Starting neural network training...")
trainer.train()

# --- 6. SAVE THE FINAL WEIGHTS ---
print(f"Saving fine-tuned model and tokenizer to {OUTPUT_MODEL_DIR}...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("Training complete! You can now upload the 'akkadian_saved_model' folder to Kaggle.")