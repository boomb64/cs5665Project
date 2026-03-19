import pandas as pd
import torch
import numpy as np
import sacrebleu
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# --- 1. LOCAL FILE PATHS ---
TRAIN_PATH = "train.csv"
OUTPUT_MODEL_DIR = "./akkadian_saved_model"
# If you downloaded the model locally in the last step, change this to "./base_byt5_model"
MODEL_NAME = "google/byt5-small"

# --- 2. LOAD & PREPARE DATA ---
print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
dataset = Dataset.from_pandas(train_df)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)

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
tokenized_datasets = split_dataset.map(preprocess_function, batched=True)


# --- 5. THE KAGGLE SCORING ALGORITHM ---
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Replace -100s (ignored tokens) with padding so the tokenizer doesn't crash
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode the AI's predictions and the actual human answer key back into English text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean up whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # SacreBLEU requires references to be a list of lists: [[ref1, ref2, ref3]]
    refs = [decoded_labels]

    # Calculate Corpus-level BLEU
    bleu_score = sacrebleu.corpus_bleu(decoded_preds, refs).score

    # Calculate Corpus-level chrF++ (word_order=2 makes it chrF++)
    chrf_score = sacrebleu.corpus_chrf(decoded_preds, refs, word_order=2).score

    # Kaggle's Geometric Mean formula
    geo_mean = np.sqrt(bleu_score * chrf_score)

    return {
        "bleu": bleu_score,
        "chrf": chrf_score,
        "geo_mean": geo_mean
    }


# --- 6. TRAIN ---
print("Configuring training parameters...")
args = Seq2SeqTrainingArguments(
    output_dir="./training_checkpoints",
    report_to="none",
    eval_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    save_strategy="epoch",

    # --- NEW: Evaluation Upgrades ---
    predict_with_generate=True,  # Forces model to generate text for the compute_metrics function
    generation_max_length=512,  # Gives it enough room to write the translation
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",  # Tell it to track your custom Kaggle score!
    greater_is_better=True,  # Unlike loss, a HIGHER Kaggle score is better
    bf16=True,  # 5070 Blackwell Superpower
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics  # <--- Plugs the custom grader into the loop
)

print("Starting neural network training with Kaggle Evaluation Metrics...")
trainer.train()

# --- 7. SAVE THE FINAL WEIGHTS ---
print(f"Saving BEST fine-tuned model to {OUTPUT_MODEL_DIR}...")
model.save_pretrained(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("Training complete! Zip the folder and upload to Kaggle.")