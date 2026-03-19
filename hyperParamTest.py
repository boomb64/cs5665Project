import pandas as pd
import torch
import itertools
import numpy as np
import sacrebleu
from datasets import Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import seaborn as sns

transformers.logging.set_verbosity_error()

# --- 1. PATHS & SETTINGS ---
TRAIN_PATH = "train.csv"
MODEL_PATH = "./akkadian_saved_model"
SAMPLE_SIZE = 150  # We only test on 150 sentences so the search takes minutes, not hours

# The exact parameters we want to test
SEARCH_SPACE = {
    "num_beams": [3, 5, 7],  # How many paths to explore
    "length_penalty": [0.8, 1.0, 1.2, 1.5],  # < 1.0 favors short text, > 1.0 favors long text
    "repetition_penalty": [1.0, 1.1, 1.2],  # 1.0 is off, higher penalizes repeating words
    "no_repeat_ngram_size": [0, 2, 3]  # 0 is off, 2 blocks repeating pairs, 3 blocks triplets
}

# --- 2. RECREATE THE QUIZ SET ---
print("Loading data and recreating the exact evaluation split...")
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
dataset = Dataset.from_pandas(train_df)

# Must use the exact same test_size and seed as your training script!
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = split_dataset["test"].select(range(SAMPLE_SIZE))

test_texts = eval_dataset["transliteration"]
true_translations = eval_dataset["translation"]

# --- 3. LOAD MODEL ---
print(f"Loading trained model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH,torch_dtype=torch.bfloat16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Add the prompt we used during training
prompted_texts = ["Translate Akkadian to English: " + text for text in test_texts]


# --- 4. THE KAGGLE GRADER ---
def score_predictions(predictions, references):
    refs = [references]
    bleu = sacrebleu.corpus_bleu(predictions, refs).score
    chrf = sacrebleu.corpus_chrf(predictions, refs, word_order=2).score
    return np.sqrt(bleu * chrf)

# --- 5. GRID SEARCH LOOP ---
keys, values = zip(*SEARCH_SPACE.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"\nStarting Grid Search: Testing {len(combinations)} parameter combinations...\n")

best_score = 0
best_params = None
results = []  # <--- We will store every single score here now
batch_size = 64

for idx, params in enumerate(combinations):
    print(f"Trial {idx + 1}/{len(combinations)} | Testing: {params}")

    predictions = []

    for i in range(0, len(prompted_texts), batch_size):
        batch = prompted_texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                early_stopping=True,
                **params
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend([pred.strip() for pred in decoded])

    # Grade this combination
    score = score_predictions(predictions, true_translations)
    print(f"--> Kaggle Score: {score:.2f}\n")

    # Save the results to our list for graphing
    results.append({**params, 'score': score})

    if score > best_score:
        best_score = score
        best_params = params

# --- 6. RESULTS & GRAPHING ---
print("=" * 50)
print(f"WINNING KAGGLE SCORE: {best_score:.2f}")
print("BEST PARAMETERS:")
for k, v in best_params.items():
    print(f"  {k} = {v}")
print("=" * 50)

print("Generating hyperparameter landscape graphs...")

# Convert results to a DataFrame
df = pd.DataFrame(results)

# Set up a 2x2 grid for our 4 parameters
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Hyperparameter Optima Landscapes", fontsize=18, fontweight='bold', y=0.98)

# The 4 parameters we want to graph
params_to_plot = ["num_beams", "length_penalty", "repetition_penalty", "no_repeat_ngram_size"]

for i, param in enumerate(params_to_plot):
    row = i // 2
    col = i % 2
    ax = axes[row, col]

    # We use a boxplot to show the spread of scores for each parameter value
    sns.boxplot(data=df, x=param, y="score", ax=ax, palette="viridis")

    # Add a line connecting the maximum values to easily spot the "peak"
    max_vals = df.groupby(param)['score'].max()
    ax.plot(range(len(max_vals)), max_vals.values, color='red', marker='o', linewidth=2, label='Max Score')

    ax.set_title(f"Effect of {param}", fontsize=14)
    ax.set_ylabel("Kaggle Score (Geo Mean)")
    ax.set_xlabel(param)
    if i == 0:
        ax.legend()

plt.tight_layout()
# Save the graph directly to your project folder
plt.savefig("hyperparameter_optima.png", dpi=300)
print("Graph saved successfully as 'hyperparameter_optima.png'!")