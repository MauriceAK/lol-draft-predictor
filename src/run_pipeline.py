import os
import argparse
import pandas as pd
from data_processing import process_raw_data, create_ml_features

def save_to_csv(dataframe: pd.DataFrame, output_path: str):
    """Saves a pandas DataFrame to a CSV file."""
    if dataframe.empty:
        print(f"Warning: DataFrame is empty. Cannot save to {output_path}.")
        return
    try:
        dataframe.to_csv(output_path, index=False)
        print(f"Successfully saved data to {output_path}")
    except Exception as e:
        print(f"An error occurred while saving the file to {output_path}: {e}")

def main(raw_dir, processed_dir):
    """
    Main function to run the full data processing pipeline.
    """
    print(f"--- Running Full Data Pipeline ---")
    os.makedirs(processed_dir, exist_ok=True)

    # --- Configuration ---
    try:
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in '{raw_dir}'. Exiting.")
            return
    except FileNotFoundError:
        print(f"Error: The directory '{raw_dir}' was not found.")
        return

    files_to_process = [os.path.join(raw_dir, fname) for fname in csv_files]
    columns_to_include = [
        'gameid', 'league', 'year', 'patch', 'datacompleteness', 'side', 'teamname', 'playername',
        'champion', 'result', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5'
    ]

    # --- Step 1: Process Raw Data into Game-per-Row Format ---
    processed_data = process_raw_data(files_to_process, columns_to_include)
    if processed_data.empty:
        print("Halting pipeline because initial processing failed.")
        return
    
    # Save the intermediate processed file for all regions
    save_to_csv(processed_data, os.path.join(processed_dir, 'all_regions_processed.csv'))

    # --- Step 2: Create ML Features for "All Regions" ---
    print("\n--- Creating ML Features for All Regions ---")
    all_regions_features = create_ml_features(processed_data)
    save_to_csv(all_regions_features, os.path.join(processed_dir, 'all_regions_ml_features.csv'))

    # --- Step 3: Filter for Main Regions and Create ML Features ---
    print("\n--- Creating ML Features for Main Regions ---")
    main_regions = ['LCK', 'LPL', 'LEC', 'LCS']
    main_regions_data = processed_data[processed_data['league'].isin(main_regions)].copy()
    
    if not main_regions_data.empty:
        # Save the intermediate processed file for main regions
        save_to_csv(main_regions_data, os.path.join(processed_dir, 'main_regions_processed.csv'))
        
        main_regions_features = create_ml_features(main_regions_data)
        save_to_csv(main_regions_features, os.path.join(processed_dir, 'main_regions_ml_features.csv'))
    else:
        print("No games found for main regions. Skipping main region feature creation.")

    print("\n--- Pipeline Finished Successfully ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the LoL data processing pipeline.")
    parser.add_argument('--input', default='data/raw', help="Path to the raw data directory.")
    parser.add_argument('--output', default='data/processed', help="Path to the processed data directory.")
    args = parser.parse_args()
    main(raw_dir=args.input, processed_dir=args.output)
