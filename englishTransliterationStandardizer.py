import pandas as pd
import re

# --- 1. LOAD DATA ---
FILE_PATH = "training_ready_master.csv"
print(f"Loading {FILE_PATH}...")
df = pd.read_csv(FILE_PATH).dropna(subset=['translation'])


# --- 2. PROGRESSIVE SCRUB ENGINE ---
def clean_english_target(text):
    text = str(text)

    # 1. Remove Meaning Unknowns
    text = re.sub(r'\badj\.?\s*mng\.?\s*unkn\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmng\.?\s*unkn\.?', '', text, flags=re.IGNORECASE)

    # 2. Remove grammar instruction parentheticals e.g., (= acc. or ana, - iš)
    text = re.sub(r'\(\=.*?\)', '', text)

    # 3. Smarter Sumerogram Nuke
    # Deletes any bracket that contains multiple uppercase letters or capital acronyms
    text = re.sub(r'\[[^\]]*?(?:[A-ZŠṢṬḪÚÍÁÉ]{2,}|[A-ZŠṢṬḪÚÍÁÉ]\.)[^\]]*?\]', '', text)

    # 4. Remove dialects, locations, and dictionary sources
    dialects = [
        r'\b[OoSsMmNnJj]+/[A-Za-z]+\b',
        r'\b[OMNjS]?[A-Za-z]\([A-Za-z]+\.?\)',
        r'\b[OMNjS]?[AB]\b',
        r'\bOAkk\b', r'\bNuzi\b', r'\bMari\b', r'\bEšn\.\b',
        r'\bBogh\.(?!\w)', r'\bUgar\.(?!\w)', r'\bUg\.(?!\w)', r'\bAm\.(?!\w)',
        r'\bBab\.(?!\w)', r'\bAss\.(?!\w)', r'\blex\.(?!\w)', r'\bvar\.(?!\w)'
    ]
    for d in dialects:
        text = re.sub(d, '', text)

    # 5. Remove Verb Stems and Grammar shorthand
    grammar = [
        r'\b[GDŠN]tn\b', r'\b[GDŠN]t\b', r'\b[GDŠN]\b', r'\bŠD\b',
        r'\biter\.?(?!\w)', r'\bstat\.?(?!\w)', r'\bellipt\.?(?!\w)',
        r'\bmed\.?(?!\w)', r'\bpass\.?(?!\w)', r'\bcaus\.?(?!\w)',
        r'\besp\.?(?!\w)', r'\bsubj\.?(?!\w)', r'\bmath\.?(?!\w)',
        r'\bpl\.?(?!\w)', r'\bf\.?(?!\w)', r'\bom\.?(?!\w)'
    ]
    for g in grammar:
        text = re.sub(g, '', text)

    # 6. Clean up the actual formatting (quotes, brackets, ellipses)
    text = text.replace('"', '')
    text = re.sub(r'[\[\]<>]', '', text)
    text = text.replace("...", " ").replace(". . .", " ")

    # 7. Punctuation and Space Wreckage Cleanup
    text = re.sub(r'\(\s*\)', '', text)  # Delete empty parentheses
    text = re.sub(r'\)\s*,', ') ', text)
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r',+', ',', text)  # Catch-all for any double commas
    text = re.sub(r'\(\s*,', '(', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = re.sub(r'[,.;:\-\s]+$', '', text)
    text = re.sub(r'^[,.;:\-\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --- 3. RANDOM SAMPLE STRESS TEST ---
print("Pulling 20 random rows from the actual dataset...")

# Grab 20 random rows
random_sample_df = df.sample(n=20).copy()

# Apply the cleaner
random_sample_df['clean_translation'] = random_sample_df['translation'].apply(clean_english_target)

print("\n--- RANDOM SAMPLE RESULTS ---")
for i, row in random_sample_df.iterrows():
    print(f"OLD: {row['translation']}")
    print(f"NEW: {row['clean_translation']}\n")

# --- 4. APPLY TO FULL DATASET (DISABLED) ---
print("Applying standardization to the full dataset...")
df['clean_translation'] = df['translation'].apply(clean_english_target)

# Save the finalized dataset
df['translation'] = df['clean_translation']
df[['transliteration', 'translation']].to_csv("training_ready_master_clean.csv", index=False)
print("Success! Cleaned data saved to 'training_ready_master_clean.csv'")