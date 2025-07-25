import os
import argparse
from data_processing import process_lol_esports_data, create_ml_features, save_to_csv

def main(raw_dir, processed_dir):
    """
    Main function to run the data processing pipeline.
    Accepts directories as arguments for better automation.
    """
    print(f"--- Running Data Pipeline ---")
    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {processed_dir}")

    os.makedirs(processed_dir, exist_ok=True)

    try:
        all_files_in_raw = os.listdir(raw_dir)
        csv_files = [f for f in all_files_in_raw if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: The directory '{raw_dir}' was not found.")
        return

    if not csv_files:
        print(f"No CSV files found in '{raw_dir}'. Exiting.")
        return

    files_to_process = [os.path.join(raw_dir, fname) for fname in csv_files]
    columns_to_include = [
        'gameid', 'league', 'year', 'patch', 'datacompleteness', 'side', 'teamname', 'playername',
        'champion', 'result', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5'
    ]

    # --- Step 1: Initial Data Processing ---
    all_data = process_lol_esports_data(files_to_process, columns_to_include)
    if all_data.empty:
        return

    # --- Step 2: Process "All Regions" Data ---
    print("\n--- Processing for All Regions ---")
    all_regions_features = create_ml_features(all_data)
    save_to_csv(all_regions_features, os.path.join(processed_dir, 'all_regions_ml_features.csv'))

    # --- Step 3: Filter and Process "Main Regions" Data ---
    print("\n--- Processing for Main Regions ---")
    main_regions = ['LCK', 'LPL', 'LEC', 'LCS']
    main_regions_data = all_data[all_data['league'].isin(main_regions)].copy()
    if not main_regions_data.empty:
        main_regions_features = create_ml_features(main_regions_data)
        save_to_csv(main_regions_features, os.path.join(processed_dir, 'main_regions_ml_features.csv'))
    else:
        print("No games found for main regions.")

    print("\n--- Pipeline Finished Successfully ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the LoL data processing pipeline.")
    parser.add_argument('--input', default='data/raw', help="Path to the raw data directory.")
    parser.add_argument('--output', default='data/processed', help="Path to the processed data directory.")
    args = parser.parse_args()
    main(raw_dir=args.input, processed_dir=args.output)