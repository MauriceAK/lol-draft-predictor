import os
import glob
import pandas as pd

from data_processing import (
    process_raw_data,
    _precompute_stats,
    transform_data_for_nn,
    create_feature_mappings,
)
from split_data import split_train_test

print("--- Running Full Data Pipeline ---")

# Load and merge all CSVs from data/raw/
raw_data_dir = 'data/raw'
input_files = glob.glob(os.path.join(raw_data_dir, "*.csv"))
required_columns = ['gameid', 'side', 'position', 'champion', 'playername', 'teamname', 'result']

# Step 1: Load and filter
df = process_raw_data(input_files, required_columns)
print(f"Rows after filtering: {len(df)}")

# Step 2: Pre-compute patch-wise statistics
stats = _precompute_stats(df)

# Step 3: Transform data into ML-ready format
print("Transforming data into pivoted NN-ready format...")
mappings = create_feature_mappings(df, 'data/processed/feature_mappings.json')
pivoted = transform_data_for_nn(df, mappings)
print(f"Shape after join: {pivoted.shape}")

# Step 4: Save processed DataFrame
pivoted.to_csv("data/processed/all_regions_processed.csv", index=False)
print("Successfully saved data to data/processed/all_regions_processed.csv")

# Step 5: Split and save train/test sets
split_train_test(pivoted, output_dir="data")
