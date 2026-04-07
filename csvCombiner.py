import pandas as pd

# 1. Load the master training data we built earlier
# (This contains the original train.csv + the 4,770 scraped rows)
train_df = pd.read_csv("training_ready_final.csv")

# 2. Load the REAL dictionary file
dict_df = pd.read_csv("eBL_Dictionary.csv")

# 3. Drop rows with missing words or definitions
dict_df = dict_df.dropna(subset=['word', 'definition'])

# 4. Rename the columns to perfectly match the training script
dict_df = dict_df.rename(columns={
    'word': 'transliteration',
    'definition': 'translation'
})

# 5. Keep only the essential columns
dict_df = dict_df[['transliteration', 'translation']]

print(f"Adding {len(dict_df)} vocabulary words from the eBL Dictionary...")

# 6. Combine them into one massive dataset
master_df = pd.concat([train_df, dict_df], ignore_index=True)

# 7. Shuffle the dataset so the dictionary words are mixed in with the sentences
master_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the new, expanded master file!
output_name = "training_ready_master.csv"
master_df.to_csv(output_name, index=False)

print(f"Success! Master training file now has {len(master_df)} rows and is saved as '{output_name}'.")