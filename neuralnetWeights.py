import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# --- 1. LOCAL FILE PATHS ---
TRAIN_PATH = "train.csv"
OUTPUT_MODEL_DIR = "./akkadian_saved_model"

# --- 2. LOAD & PREPARE DATA ---
print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
dataset = Dataset.from_pandas(train_df)

# NEW: Split the data so the model has a 10% "quiz" set to evaluate itself on
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

# --- 3. LOAD BASE MODEL ---
print("Downloading base ByT5 model from Hugging Face...")
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- QUICK FIX FOR WINDOWS TOKENIZER HANG ---
cached_vocab = {k.content: v for v, k in sorted(tokenizer._added_tokens_decoder.items(), key=lambda item: item[0])}
type(tokenizer).added_tokens_encoder = property(lambda self: cached_vocab)


# --- 4. PREPROCESSING ---
def preprocess_function(examples):
    # NEW: Add a strict prompt so the model knows exactly what task to perform
    inputs = ["Translate Akkadian to English: " + str(ex) for ex in examples["transliteration"]]

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["translation"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing data...")
tokenized_datasets = split_dataset.map(preprocess_function, batched=True)

# --- 5. TRAIN ---
print("Configuring training parameters...")
args = Seq2SeqTrainingArguments(
    output_dir="./training_checkpoints",
    report_to="none",
    eval_strategy="epoch",  # NEW: Check performance after every epoch
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=15,  # NEW: Massive increase in training time
    save_strategy="epoch",  # NEW: Save a checkpoint every epoch
    load_best_model_at_end=True,  # NEW: Keep only the smartest version
    bf16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],  # The 90% training data
    eval_dataset=tokenized_datasets["test"],  # The 10% quiz data
    processing_class=tokenizer,
)

print("Starting neural network training...")
trainer.train()

# --- 6. SAVE THE FINAL WEIGHTS ---
print(f"Saving fine-tuned model and tokenizer to {OUTPUT_MODEL_DIR}...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("Training complete! Zip the folder and upload to Kaggle.")