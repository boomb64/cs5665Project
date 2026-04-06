import pandas as pd

# --- 1. LOAD DATA ---
TRAIN_PATH = "train.csv"
print(f"Loading dataset from {TRAIN_PATH}...\n")
df = pd.read_csv(TRAIN_PATH)

# --- 2. BASIC OVERVIEW ---
print("=== DATASET OVERVIEW ===")
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"Columns: {list(df.columns)}")

print("\n=== MISSING VALUES ===")
# This tells you how many broken/empty rows you have
print(df.isnull().sum())

# Drop missing values so our math below doesn't crash
df = df.dropna(subset=['transliteration', 'translation'])

# --- 3. CHARACTER COUNT STATISTICS ---
# ByT5 is a byte-level model, so character counts are incredibly important!
df['akk_char_len'] = df['transliteration'].apply(lambda x: len(str(x)))
df['eng_char_len'] = df['translation'].apply(lambda x: len(str(x)))

print("\n=== CHARACTER COUNT (ByT5 Tokens) ===")
print("Akkadian (Transliteration):")
# The 99% percentile is the magic number for setting max_length!
print(df['akk_char_len'].describe(percentiles=[.50, .75, .90, .95, .99]).round(1))

print("\nEnglish (Translation):")
print(df['eng_char_len'].describe(percentiles=[.50, .75, .90, .95, .99]).round(1))

# --- 4. WORD COUNT STATISTICS ---
df['akk_word_count'] = df['transliteration'].apply(lambda x: len(str(x).split()))
df['eng_word_count'] = df['translation'].apply(lambda x: len(str(x).split()))

print("\n=== WORD COUNT ===")
print("Akkadian Words per Sentence:")
print(df['akk_word_count'].describe()[['mean', 'min', 'max']].round(1))

print("\nEnglish Words per Sentence:")
print(df['eng_word_count'].describe()[['mean', 'min', 'max']].round(1))
print("=======================\n")