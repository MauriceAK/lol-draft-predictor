# scripts/predict_batch.py

import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from lolpredictor.etl import build_team_vector
import config

def predict_batch_with_accuracy(csv_filepath):
    """
    Loads a trained model, predicts outcomes for all games in a given CSV file,
    compares them to the actual results, and reports overall accuracy.
    Filters for games only in specified leagues.
    """
    print("Loading model and necessary files...")
    MODEL_PATH = f"{config.MODELS_DIR}/01_siamese_model.keras"
    MAPS_PATH = f"{config.MODELS_DIR}/feature_maps.pkl"
    FEATURES_PATH = f"{config.MODELS_DIR}/feature_names.pkl"
    TEAM_MAPPING_PATH = "scripts/team_mapping.json"
    SCALER1_PATH = f"{config.MODELS_DIR}/scaler1.pkl"
    SCALER2_PATH = f"{config.MODELS_DIR}/scaler2.pkl"

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(MAPS_PATH, 'rb') as f: maps = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f: feature_names = pickle.load(f)
        with open(TEAM_MAPPING_PATH, 'r') as f: team_mapping = json.load(f)
        with open(SCALER1_PATH, 'rb') as f: scaler1 = pickle.load(f)
        with open(SCALER2_PATH, 'rb') as f: scaler2 = pickle.load(f)
    except FileNotFoundError as e:
        print(f"ERROR: A necessary model file was not found: {e.filename}")
        print("Please ensure you have run train.py to generate the model and associated files.")
        return

    print(f"Reading and processing new games from: {csv_filepath}\n")
    try:
        new_games_df = pd.read_csv(csv_filepath, dtype={'gameid': str})
    except FileNotFoundError:
        print(f"ERROR: The file was not found at the specified path: {csv_filepath}")
        return

    # --- Define the list of leagues to include ---
    allowed_leagues = {'LCS', 'LCK', 'LPL', 'LEC', 'LTA S', 'LTA N', 'LCP', 'CBLOL', 'MSI', 'WLDs', 'EWC'}

    total_games_processed = 0
    correct_predictions = 0
    
    role_order = ['top', 'jng', 'mid', 'bot', 'sup']
    
    for game_id, game_df in new_games_df.groupby('gameid'):
        player_rows = game_df[game_df['position'].notna()].copy()
        if len(player_rows) < 10:
            continue

        # --- NEW: Filter games based on league ---
        game_league = player_rows['league'].iloc[0]
        if game_league not in allowed_leagues:
            continue
        # --- End of new section ---

        patch = player_rows['patch'].iloc[0]
        blue_players = player_rows[player_rows['side'] == 'Blue']
        red_players = player_rows[player_rows['side'] == 'Red']
        
        blue_team_name = blue_players['teamname'].iloc[0]
        red_team_name = red_players['teamname'].iloc[0]

        blue_team_id = team_mapping.get(blue_team_name)
        red_team_id = team_mapping.get(red_team_name)

        if not blue_team_id or not red_team_id:
            continue

        total_games_processed += 1

        blue_picks_dict = blue_players.set_index('position')['champion'].to_dict()
        blue_pids_dict = blue_players.set_index('position')['playerid'].to_dict()
        red_picks_dict = red_players.set_index('position')['champion'].to_dict()
        red_pids_dict = red_players.set_index('position')['playerid'].to_dict()

        blue_picks = [blue_picks_dict.get(role) for role in role_order]
        blue_pids = [blue_pids_dict.get(role) for role in role_order]
        red_picks = [red_picks_dict.get(role) for role in role_order]
        red_pids = [red_pids_dict.get(role) for role in role_order]

        v_blue = build_team_vector(blue_team_id, 0, patch, blue_picks, red_picks, blue_pids, maps)
        v_red = build_team_vector(red_team_id, 0, patch, red_picks, blue_picks, red_pids, maps)

        v_blue_ordered = [v_blue.get(fname, 0.0) for fname in feature_names]
        v_red_ordered = [v_red.get(fname, 0.0) for fname in feature_names]

        X_blue_raw = np.array([v_blue_ordered], dtype=np.float32)
        X_red_raw = np.array([v_red_ordered], dtype=np.float32)

        X_blue_raw = np.nan_to_num(X_blue_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_red_raw = np.nan_to_num(X_red_raw, nan=0.0, posinf=0.0, neginf=0.0)

        X_blue_scaled = scaler1.transform(X_blue_raw)
        X_red_scaled = scaler2.transform(X_red_raw)

        win_prob_blue = model.predict([X_blue_scaled, X_red_scaled], verbose=0)[0][0]
        win_prob_red = 1 - win_prob_blue
        
        predicted_winner = "Blue" if win_prob_blue > 0.5 else "Red"
        actual_winner_label = blue_players['result'].iloc[0]
        actual_winner = "Blue" if actual_winner_label == 1 else "Red"

        is_correct = (predicted_winner == actual_winner)
        if is_correct:
            correct_predictions += 1

        print(f"--- Prediction for Game ID: {game_id} (League: {game_league}) ---")
        print(f"  Blue Side ({blue_team_name}): {win_prob_blue:.1%}")
        print(f"  Red Side  ({red_team_name}): {win_prob_red:.1%}")
        print(f"  Actual Winner: {actual_winner} Side ({blue_team_name if actual_winner == 'Blue' else red_team_name})")
        print(f"  Prediction was {'CORRECT' if is_correct else 'WRONG'}")
        print("-" * (46 + len(game_id) + len(game_league)) + "\n")

    if total_games_processed > 0:
        accuracy = (correct_predictions / total_games_processed) * 100
        print("\n" + "="*40)
        print("           FINAL ACCURACY REPORT")
        print("="*40)
        print(f"  Leagues Processed: {', '.join(allowed_leagues)}")
        print(f"  Total Games Processed: {total_games_processed}")
        print(f"  Correct Predictions:   {correct_predictions}")
        print(f"  Model Accuracy:        {accuracy:.2f}%")
        print("="*40)
    else:
        print("No valid games were processed from the CSV file for the specified leagues.")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scripts/predict_batch.py <path_to_your_csv_file>")
    else:
        csv_path = sys.argv[1]
        predict_batch_with_accuracy(csv_path)