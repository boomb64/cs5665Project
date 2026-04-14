import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter

# ==========================================
# 1. LOCAL FILE PATHS
# ==========================================
ORIGINAL_TRAIN_PATH = "train.csv"
SCRAPED_TRAIN_PATH = "training_ready_final.csv"


# ==========================================
# 2. THE PREPROCESSING ENGINE
# ==========================================
def normalize_akkadian(text):
    """Standardizes Akkadian characters."""
    text = str(text).lower()
    text = text.replace("sz", "š").replace("s,", "ṣ").replace("t,", "ṭ").replace("h", "ḫ")
    # Fix smart quotes and em-dashes
    text = text.replace('”', '"').replace('“', '"').replace('—', '-')
    return text


def standardize_gaps(text):
    """Standardizes missing text into a uniform [GAP] token."""
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


# ==========================================
# 3. CHART GENERATOR
# ==========================================
def plot_vocab_frequency(df, column_name, top_n=25):
    """Calculates word frequencies and generates a bar chart."""
    print(f"Calculating vocabulary frequencies for: {column_name}...")

    # Combine all text and split into words based on whitespace
    all_text = ' '.join(df[column_name].astype(str).tolist())
    words = all_text.split()

    # Count frequencies
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)

    # Unpack for plotting
    vocab, counts = zip(*most_common)

    # Build the chart
    plt.figure(figsize=(14, 6))

    # Use different colors to easily distinguish the two charts
    bar_color = '#4C72B0' if column_name == 'transliteration' else '#55A868'

    plt.bar(vocab, counts, color=bar_color, edgecolor='black')
    plt.title(f'Top {top_n} Most Frequent Words in {column_name.capitalize()}', fontsize=14, fontweight='bold')
    plt.xlabel('Vocabulary Token', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Add exact numbers on top of the bars
    for i, count in enumerate(counts):
        plt.text(i, count + (max(counts) * 0.01), str(count), ha='center', fontsize=9)

    plt.tight_layout()
    output_filename = f"{column_name}_vocab_frequency.png"
    plt.savefig(output_filename, dpi=300)
    print(f"-> Saved chart as {output_filename}")
    plt.show()


# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Loading high-quality training datasets...")
    try:
        df_original = pd.read_csv(ORIGINAL_TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
        df_scraped = pd.read_csv(SCRAPED_TRAIN_PATH).dropna(subset=['transliteration', 'translation'])
    except FileNotFoundError as e:
        print(
            f"Error: Could not find data files. Ensure {ORIGINAL_TRAIN_PATH} and {SCRAPED_TRAIN_PATH} are in this directory.")
        exit()

    # Combine and deduplicate
    master_df = pd.concat([df_original, df_scraped], ignore_index=True)
    master_df = master_df.drop_duplicates(subset=['transliteration'])
    print(f"Dataset loaded: {len(master_df)} pristine sentences.")

    # Apply standardization
    print("Standardizing characters and gaps...")
    master_df['transliteration'] = master_df['transliteration'].apply(normalize_akkadian).apply(standardize_gaps)
    master_df['translation'] = master_df['translation'].apply(standardize_gaps)

    # Generate the plots
    plot_vocab_frequency(master_df, 'transliteration', top_n=25)
    plot_vocab_frequency(master_df, 'translation', top_n=25)

    print("Done!")