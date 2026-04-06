import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

# --- 1. LOAD AND CLEAN DATA ---
print("Loading published_texts.csv...")
df = pd.read_csv("published_texts.csv")

# We need rows with an ID and Transliteration
df = df.dropna(subset=['cdli_id', 'transliteration'])

# Clean IDs (Fixes the "P504745 | P290300" issue by grabbing just the first ID)
df['clean_id'] = df['cdli_id'].astype(str).apply(lambda x: x.split('|')[0].strip())

print(f"Found {len(df)} texts to process. Starting high-speed fetch...\n")

# --- 2. SETUP CACHE ---
# We will store downloaded buckets here so we don't download 'p361.json' 500 times.
bucket_cache = {}
translations = []

# --- 3. FETCH AND PARSE DATA ---
for index, row in df.iterrows():
    tablet_id = row['clean_id']

    # Get the 4-character prefix (e.g., 'P361099' -> 'p361')
    prefix = tablet_id[:4].lower()

    # Check if we already downloaded this bucket
    if prefix not in bucket_cache:
        url = f"https://aicuneiform.com/p/{prefix}.json"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                bucket_cache[prefix] = response.json()
            else:
                bucket_cache[prefix] = None  # Mark as failed so we don't retry endlessly
        except Exception as e:
            bucket_cache[prefix] = None

        time.sleep(0.1)  # Tiny pause only when downloading a new bucket

    # --- EXTRACT THE HTML ---
    english_text = "MISSING: NO TRANSLATION FOUND"
    bucket_data = bucket_cache.get(prefix)

    if bucket_data and tablet_id in bucket_data:
        html_string = bucket_data[tablet_id].get('html', '')

        if html_string:
            # Parse the hidden HTML
            soup = BeautifulSoup(html_string, 'html.parser')

            # Find all English sections (this captures Obverse, Reverse, Edges, etc.)
            english_divs = soup.find_all('div', class_='lang-ml_en')

            if english_divs:
                tablet_parts = []
                for div in english_divs:
                    # Target just the <p> tags so we don't grab the "AI Translation" label
                    p_tags = div.find_all('p')
                    for p in p_tags:
                        tablet_parts.append(p.get_text(separator=' ', strip=True))

                # Combine Obverse and Reverse into one clean sentence
                if tablet_parts:
                    english_text = " ".join(tablet_parts)

    translations.append(english_text)

    # --- LIVE CHECK (Every 50 rows because it will move fast) ---
    if len(translations) % 50 == 0:
        print(f"Progress: {len(translations)} / {len(df)}")
        preview = english_text[:120] + "..." if len(english_text) > 120 else english_text
        print(f"[{tablet_id}] -> {preview}\n")

        # Save checkpoint
        temp_df = df.iloc[:len(translations)].copy()
        temp_df['translation'] = translations
        temp_df[['transliteration', 'translation']].to_csv("checkpoint_fast.csv", index=False)

# --- 4. CLEANUP AND FINAL SAVE ---
df['translation'] = translations

failed_df = df[df['translation'].str.startswith('MISSING')]
print(f"\nFetch complete. {len(failed_df)} tablets did not have a translation available.")

clean_df = df[~df['translation'].str.startswith('MISSING')]

output_filename = "training_ready_final.csv"
clean_df[['transliteration', 'translation']].to_csv(output_filename, index=False)

print(f"Successfully saved {len(clean_df)} perfect translations to {output_filename}!")