import os
from data_processing import process_lol_esports_data, create_ml_features, save_to_csv

def main():
    """
    Main function to run the data processing pipeline.
    """
    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')

    os.makedirs(processed_data_dir, exist_ok=True)

    try:
        all_files_in_raw = os.listdir(raw_data_dir)
        csv_files = [f for f in all_files_in_raw if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: The directory '{raw_data_dir}' was not found. Please make sure it exists.")
        return

    if not csv_files:
        print(f"No CSV files found in '{raw_data_dir}'. Exiting.")
        return

    print(f"Found the following files to process: {csv_files}")
    
    files_to_process = [os.path.join(raw_data_dir, fname) for fname in csv_files]

    columns_to_include = [
        'gameid', 'league', 'year', 'patch', 'datacompleteness', 'side', 'teamname', 'playername',
        'champion', 'result', 'ban1', 'ban2', 'ban3', 'ban4', 'ban5'
    ]
    
    # --- Step 1: Initial Data Processing ---
    all_data = process_lol_esports_data(files_to_process, columns_to_include)

    if all_data.empty:
        return # Exit if initial processing failed

    # --- Step 2: Process "All Regions" Data ---
    print("\n--- Processing for All Regions ---")
    all_regions_output_filename = 'all_regions_processed.csv'
    all_regions_output_filepath = os.path.join(processed_data_dir, all_regions_output_filename)
    save_to_csv(all_data, all_regions_output_filepath)
    print(f"Successfully saved all {len(all_data)} games to {all_regions_output_filepath}")

    # Create ML features for ALL regions
    all_regions_ml_features_filename = 'all_regions_ml_features.csv'
    all_regions_ml_features_filepath = os.path.join(processed_data_dir, all_regions_ml_features_filename)
    all_regions_features_df = create_ml_features(all_data)
    
    if not all_regions_features_df.empty:
        save_to_csv(all_regions_features_df, all_regions_ml_features_filepath)
        print(f"Successfully created and saved {len(all_regions_features_df)} model-ready games for ALL regions to {all_regions_ml_features_filepath}")


    # --- Step 3: Filter and Process "Main Regions" Data ---
    print("\n--- Processing for Main Regions ---")
    main_regions = ['LCK', 'LPL', 'LEC', 'LCS']
    
    if 'league' in all_data.columns:
        main_regions_data = all_data[all_data['league'].isin(main_regions)].copy()
        
        if not main_regions_data.empty:
            # Save the intermediate processed file
            main_regions_output_filename = 'main_regions_processed.csv'
            main_regions_output_filepath = os.path.join(processed_data_dir, main_regions_output_filename)
            save_to_csv(main_regions_data, main_regions_output_filepath)
            print(f"Successfully saved {len(main_regions_data)} main region games to {main_regions_output_filepath}")

            # Create ML features for MAIN regions
            main_regions_ml_features_filename = 'main_regions_ml_features.csv'
            main_regions_ml_features_filepath = os.path.join(processed_data_dir, main_regions_ml_features_filename)
            main_regions_features_df = create_ml_features(main_regions_data)

            if not main_regions_features_df.empty:
                save_to_csv(main_regions_features_df, main_regions_ml_features_filepath)
                print(f"Successfully created and saved {len(main_regions_features_df)} model-ready games for MAIN regions to {main_regions_ml_features_filepath}")
        else:
            print("No games found for the specified main regions.")
    else:
        print("Warning: 'league' column not found in the processed data. Cannot filter for main regions.")


if __name__ == '__main__':
    main()