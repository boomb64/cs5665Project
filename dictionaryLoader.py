import pandas as pd
import re

# --- 1. LOAD DATA ---
FILE_PATH = "eBL_Dictionary.csv"
print(f"Loading {FILE_PATH}...")
df = pd.read_csv(FILE_PATH).dropna(subset=['word', 'definition'])

# --- 2. PROGRESSIVE SCRUB ENGINE ---
def clean_english_target(text):
    text = str(text)

    # 1. Remove Meaning Unknowns
    text = re.sub(r'\badj\.?\s*mng\.?\s*unkn\.?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmng\.?\s*unkn\.?', '', text, flags=re.IGNORECASE)

    # 2. Remove grammar instruction parentheticals e.g., (= acc. or ana, - iš)
    text = re.sub(r'\(\=.*?\)', '', text)

    # 3. Smarter Sumerogram Nuke
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
        r'\bpl\.?(?!\w)', r'\bf\.?(?!\w)', r'\bom\.?(?!\w)',
        r'\bcf\.?(?!\w)' # Added "cf." just in case!
    ]
    for g in grammar:
        text = re.sub(g, '', text)

    # 6. Clean up the actual formatting (quotes, brackets, ellipses)
    text = text.replace('"', '')
    text = re.sub(r'[\[\]<>]', '', text)
    text = text.replace("...", " ").replace(". . .", " ")

    # 7. Punctuation and Space Wreckage Cleanup
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\)\s*,', ') ', text)
    text = re.sub(r'\s*,\s*,', ',', text)
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\(\s*,', '(', text)
    text = re.sub(r'\s+([,.;:])', r'\1', text)
    text = re.sub(r'[,.;:\-\s]+$', '', text)
    text = re.sub(r'^[,.;:\-\s]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# --- 3. APPLY TO DATASET ---
print("Applying standardization to the dictionary...")
df['clean_translation'] = df['definition'].apply(clean_english_target)

# --- 4. FORMAT FOR TWO-STAGE TRAINING ---
# We map the columns to what the training script expects!
final_df = pd.DataFrame({
    'transliteration': df['word'],
    'translation': df['clean_translation']
})

# Drop any rows that became completely empty after cleaning
final_df = final_df[final_df['translation'].str.strip() != '']

# Save the finalized dataset
output_name = "dictionary_and_fragments.csv"
final_df.to_csv(output_name, index=False)
print(f"Success! {len(final_df)} clean dictionary rows saved to '{output_name}'")