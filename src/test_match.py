# src/test_match.py

import pandas as pd
from predict import load_latest_model, predict_winner

if __name__ == '__main__':
    # --- Define the new game's raw information ---
    # The patch number is now just a string, not a one-hot encoded feature
    game_to_predict = {
        'league': 'LCK',
        'patch': '14.14', # IMPORTANT: Must be a patch that exists in your historical data
        'blue_team': 'FLY FAM',
        'red_team': 'T1',
        'blue_players': ['Zamudo', 'Will', 'Faker', 'Gumayusi', 'Keria'],
        'red_players': ['Kiin', 'Canyon', 'Chovy', 'Ruler', 'Lehends'],
        'blue_champions': ['Jayce', 'Xayah', 'Azir', 'Sejuani', 'Rakan'],
        'blue_bans': ['Hwei', 'Aurora', 'Senna', 'Xin Zhao', 'Taliyah'],
        'red_champions': ['K\'Sante', 'Lucian', 'Twisted Fate', 'Skarner', 'Braum'],
        'red_bans': ['Pantheon', 'Varus', 'Jarvan IV', 'Trundle', 'Nocturne']
    }

    # 1. Load the historical data needed to create the features
    try:
        historical_data_path = 'data/processed/main_regions_processed.csv'
        historical_df = pd.read_csv(historical_data_path)
        # We need to eval the list-like columns
        for col in ['blue_players', 'blue_champions', 'blue_bans', 'red_players', 'red_champions', 'red_bans']:
             historical_df[col] = historical_df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    except FileNotFoundError:
        print(f"Error: Historical data not found at {historical_data_path}. Please run the pipeline first.")
        exit()

    # 2. Load the latest trained model
    predictor_model = load_latest_model()
    
    # 3. Get the prediction
    if predictor_model:
        result = predict_winner(predictor_model, game_to_predict, historical_df)
        print("\n--- Match Prediction ---")
        print(f"{game_to_predict['blue_team']} (Blue) vs. {game_to_predict['red_team']} (Red)")
        print(result)
