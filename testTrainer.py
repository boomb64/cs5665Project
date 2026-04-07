import pandas as pd
import torch
import numpy as np
# import sacrebleu  # <-- Commented out as it's only needed for evaluation/tuning
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# --- 1. LOCAL FILE PATHS ---
# Now we just load the single master file that has the original data, scraped data, and dictionary
MASTER_TRAIN_PATH = "training_ready_master.csv"
OUTPUT_MODEL_DIR = "./akkadian_saved_model"
MODEL_NAME = "google/byt5-base"

# --- 2. LOAD DATA ---
print(f"Loading master training data from {MASTER_TRAIN_PATH}...")

train_df = pd.read_csv(MASTER_TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
print(f"Loaded {len(train_df)} total training rows.")

# Convert directly to Hugging Face format (No train/test split needed for pure training)
dataset = Dataset.from_pandas(train_df)

# --- 3. LOAD BASE MODEL ---
print(f"Loading base ByT5 model ({MODEL_NAME})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Fix for Windows Tokenizer Hang
cached_vocab = {k.content: v for v, k in sorted(tokenizer._added_tokens_decoder.items(), key=lambda item: item[0])}
type(tokenizer).added_tokens_encoder = property(lambda self: cached_vocab)


# --- 4. PREPROCESSING ---
def preprocess_function(examples):
    inputs = ["Translate Akkadian to English: " + str(ex) for ex in examples["transliteration"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["translation"], max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


print("Tokenizing data...")
# We map the full dataset directly since we removed the split
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# --- 5. THE KAGGLE SCORING ALGORITHM (COMMENTED OUT FOR PURE TRAINING) ---
# def compute_metrics(eval_preds):
#     ...
#     return {"bleu": bleu_score, "chrf": chrf_score, "geo_mean": geo_mean}


# --- 6. TRAIN ---
print("Configuring training parameters...")
args = Seq2SeqTrainingArguments(
    output_dir="./training_checkpoints",
    report_to="none",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    save_strategy="no",  # Don't save intermediate checkpoints to save disk space
    bf16=True,  # Keep the Blackwell GPU speedup

    # --- COMMENTED OUT TUNING & EVALUATION ---
    # eval_strategy="epoch",
    # predict_with_generate=True,
    # generation_max_length=512,
    # load_best_model_at_end=True,
    # metric_for_best_model="geo_mean",
    # greater_is_better=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,  # Training on 100% of the data
    # eval_dataset=tokenized_datasets["test"],  # Commented out
    processing_class=tokenizer,
    # compute_metrics=compute_metrics    # Commented out
    label_smoothing_factor=0.1,  # <--- ADD THIS
    warmup_ratio=0.1,            # <--- ADD THIS
    weight_decay=0.01           # <--- ADD THIS (helps prevent overfitting)
)

print("Starting full-throttle neural network training (No evaluation pauses)...")
trainer.train()

# --- 7. SAVE THE FINAL WEIGHTS ---
print(f"Saving final fine-tuned model to {OUTPUT_MODEL_DIR}...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("Training complete! Zip the folder and upload to Kaggle.")