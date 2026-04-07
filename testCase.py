import pandas as pd
import glob

# Get a list of all CSVs in the directory
csv_files = glob.glob("*.csv")

for file in csv_files:
    print(f"\n--- {file} ---")
    try:
        # We use nrows=0 to only load the headers, making it instantaneous
        df = pd.read_csv(file, nrows=0)
        print("Columns:", df.columns.tolist())
    except Exception as e:
        print(f"Could not read {file}: {e}")