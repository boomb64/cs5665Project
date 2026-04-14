import pandas as pd
import torch
import numpy as np
import evaluate
import re
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# --- 1. LOCAL FILE PATHS ---
ORIGINAL_TRAIN_PATH = "train.csv"
SCRAPED_TRAIN_PATH = "training_ready_final.csv"
OUTPUT_MODEL_DIR = "./akkadian_training_test_visualization"
MODEL_NAME = "google/byt5-small"

# --- 2. LOAD & COMBINE GOLD DATA ---
print("Loading high-quality training datasets...")
df_original = pd.read_csv(ORIGINAL_TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
df_scraped = pd.read_csv(SCRAPED_TRAIN_PATH).dropna(subset=['transliteration', 'translation'])

master_df = pd.concat([df_original, df_scraped], ignore_index=True)
master_df = master_df.drop_duplicates(subset=['transliteration'])
print(f"Combined dataset size: {len(master_df)} pristine sentences.")


# --- 3. THE PREPROCESSING ENGINE ---
def normalize_akkadian(text):
    text = str(text).lower()
    text = text.replace("sz", "š").replace("s,", "ṣ").replace("t,", "ṭ").replace("h", "ḫ")
    # Fix smart quotes and em-dashes that confuse the evaluator
    text = text.replace('”', '"').replace('“', '"').replace('—', '-')
    return text


def standardize_gaps(text):
    text = str(text)
    # 1. Replace 3 or more dots -> [GAP]
    text = re.sub(r'\[?\s*(?:\.\s*){3,}\s*\]?', ' [GAP] ', text)
    # 2. Replace sequences of x's -> [GAP]
    text = re.sub(r'\[?\s*(?:[xX]\s*)+\s*\]?', ' [GAP] ', text)
    # 3. Replace empty brackets -> [GAP]
    text = re.sub(r'\[\s*\]', ' [GAP] ', text)
    # 4. Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Consolidate multiple back-to-back [GAP]s into a single [GAP]
    text = re.sub(r'(\[GAP\]\s*)+', '[GAP] ', text).strip()
    return text


print("Standardizing characters and gaps...")
# Apply character normalization ONLY to Akkadian
master_df['transliteration'] = master_df['transliteration'].apply(normalize_akkadian).apply(standardize_gaps)

# Apply gap standardization to BOTH Akkadian and English!
master_df['translation'] = master_df['translation'].apply(standardize_gaps)

dataset = Dataset.from_pandas(master_df)

print("Splitting data into 90% Training and 10% Validation...")
split_datasets = dataset.train_test_split(test_size=0.1, seed=42)

# --- 4. PREPARATION & MODEL SETUP ---
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to("cuda")

cached_vocab = {k.content: v for v, k in sorted(tokenizer._added_tokens_decoder.items(), key=lambda item: item[0])}
type(tokenizer).added_tokens_encoder = property(lambda self: cached_vocab)


def preprocess_function(examples):
    inputs = ["Translate Akkadian to English: " + str(ex) for ex in examples["transliteration"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["translation"], max_length=512, truncation=True, padding="max_length")

    labels_with_ignore_index = []
    for label in labels["input_ids"]:
        labels_with_ignore_index.append([l if l != tokenizer.pad_token_id else -100 for l in label])

    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs


print("Tokenizing data...")
tokenized_datasets = split_datasets.map(preprocess_function, batched=True)

# --- 5. THE KAGGLE SCORING ALGORITHM ---
# Load both metrics needed for the Kaggle evaluation
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    # --- CRITICAL FIX ---
    # Strip out any rogue -100 padding tokens from the AI's predictions
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Strip out the -100 padding from the true labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Format for SacreBLEU
    decoded_labels = [[label] for label in decoded_labels]

    # Compute BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)

    # Compute chrF++
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)

    # Extract raw scores and KAGGLE FORMULA
    bleu_score = bleu_result["score"]
    chrf_score = chrf_result["score"]
    geo_mean = (bleu_score * chrf_score) ** 0.5

    return {"bleu": bleu_score, "chrf": chrf_score, "geo_mean": geo_mean}


# --- 6. THE GOLD STANDARD TRAINING RUN ---
print("Configuring training parameters...")
args = Seq2SeqTrainingArguments(
    output_dir="./visualization_checkpoints",
    report_to="none",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=50,
    fp16=False,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_steps=1000,
    weight_decay=0.01,

    # --- KAGGLE VALIDATION SETTINGS ---
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="geo_mean",  # <--- Tell HF to use our custom geometric mean!
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=256
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

# NOTE: If you are running this entirely from scratch, you can change
# `resume_from_checkpoint=True` to `resume_from_checkpoint=False`
print("Initiating Visualization Training...")
trainer.train(resume_from_checkpoint=True)

# --- 7. EXTRACT AND PLOT PER-EPOCH METRICS ---
print("Extracting training history...")
log_history = trainer.state.log_history

epochs = []
train_loss = []
eval_loss = []
eval_geo_mean = []
eval_bleu = []

train_logs = {log['epoch']: log['loss'] for log in log_history if 'loss' in log}
eval_logs = {log['epoch']: log for log in log_history if 'eval_loss' in log}

for epoch in sorted(eval_logs.keys()):
    epochs.append(epoch)
    eval_loss.append(eval_logs[epoch]['eval_loss'])

    # Extract evaluation metrics
    eval_geo_mean.append(eval_logs[epoch].get('eval_geo_mean', 0))
    eval_bleu.append(eval_logs[epoch].get('eval_bleu', 0))

    # Grab the closest training loss for this epoch
    if train_logs:
        closest_train_epoch = min(train_logs.keys(), key=lambda k: abs(k - epoch))
        train_loss.append(train_logs[closest_train_epoch])

# --- Plot 1: Training vs Validation Loss ---
print("Generating loss charts...")
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Training Loss', marker='o', color='#4C72B0')
plt.plot(epochs, eval_loss, label='Validation Loss', marker='s', color='#DD8452')
plt.title('Training vs Validation Loss per Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("loss_per_epoch.png", dpi=300)
plt.show()

# --- Plot 2: Kaggle Evaluation Metrics ---
print("Generating Kaggle metrics charts...")
plt.figure(figsize=(10, 5))
plt.plot(epochs, eval_geo_mean, label='Validation Geo Mean (Target)', marker='*', color='#55A868', markersize=10)
plt.plot(epochs, eval_bleu, label='Validation BLEU', marker='^', color='#C44E52', linestyle='--')
plt.title('Evaluation Metrics per Epoch', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("metrics_per_epoch.png", dpi=300)
plt.show()

# --- 8. SAVE OUT ---
print(f"Saving BEST model to {OUTPUT_MODEL_DIR}...")
trainer.save_model(OUTPUT_MODEL_DIR)
tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
print("Training complete! The best-performing Kaggle checkpoint has been saved.")