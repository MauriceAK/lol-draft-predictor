# scripts/predict_live.py

import json
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from lolpredictor.etl import build_team_vector, get_connection
import config

def get_last_roster(team_id, con):
    # Fixed the SQL query to use 'gamedate'
    query = """
    WITH last_game AS (
        SELECT gameid FROM raw_player_stats WHERE teamid = ? ORDER BY date DESC LIMIT 1
    )
    SELECT position, playerid FROM raw_player_stats WHERE teamid = ? AND gameid = (SELECT gameid FROM last_game);
    """
    roster_df = con.execute(query, [team_id, team_id]).df()
    if roster_df.empty: return {}
    return pd.Series(roster_df.playerid.values, index=roster_df.position).to_dict()

print("Loading model and necessary files...")
MODEL_PATH = f"{config.MODELS_DIR}/01_siamese_model.keras"
MAPS_PATH = f"{config.MODELS_DIR}/feature_maps.pkl"
FEATURES_PATH = f"{config.MODELS_DIR}/feature_names.pkl"
TEAM_MAPPING_PATH = "scripts/team_mapping.json"
SCALER1_PATH = f"{config.MODELS_DIR}/scaler1.pkl"
SCALER2_PATH = f"{config.MODELS_DIR}/scaler2.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(MAPS_PATH, 'rb') as f: maps = pickle.load(f)
with open(FEATURES_PATH, 'rb') as f: feature_names = pickle.load(f)
with open(TEAM_MAPPING_PATH, 'r') as f: team_mapping = json.load(f)
with open('scripts/draft_live.json', 'r') as f: draft = json.load(f)
with open(SCALER1_PATH, 'rb') as f: scaler1 = pickle.load(f)
with open(SCALER2_PATH, 'rb') as f: scaler2 = pickle.load(f)

con = get_connection()
teams_data = {}
role_order = ['top', 'jng', 'mid', 'bot', 'sup']

for team_color in ['blue_team', 'red_team']:
    team_info = draft[team_color]
    team_name = team_info['name']
    team_id = team_mapping.get(team_name)
    if not team_id: raise ValueError(f"Team '{team_name}' not found in mapping.")
    
    roster = get_last_roster(team_id, con)
    if not roster: raise ValueError(f"No roster found for team '{team_name}'.")

    teams_data[team_color] = {
        "id": team_id,
        "picks": [team_info['picks'].get(role) for role in role_order],
        "players": [roster.get(role) for role in role_order]
    }
con.close()

blue_data = teams_data['blue_team']
red_data = teams_data['red_team']
patch = float(draft['patch'])
print("\nDEBUG: Attempting to look up stats for this blue team player-champion combo:")
print(f"  - Player ID: {blue_data['players'][0]}")
print(f"  - Champion:  {blue_data['picks'][0]}\n")
# This now calls the clean function signature directly, which is much more robust
v_blue = build_team_vector(
    team_id=blue_data['id'],
    match_id=0, # Placeholder for a live match
    patch=patch,
    team_picks=blue_data['picks'],
    opp_picks=red_data['picks'],
    team_players=blue_data['players'],
    maps=maps
)
v_red = build_team_vector(
    team_id=red_data['id'],
    match_id=0, # Placeholder for a live match
    patch=patch,
    team_picks=red_data['picks'],
    opp_picks=blue_data['picks'],
    team_players=red_data['players'],
    maps=maps
)

print(blue_data)
print(red_data)
print("\n--- Team Vectors ---")
print("Blue Team Vector:", v_blue)
print("Red Team Vector:", v_red)

# --- Final Data Preparation and Prediction ---
v_blue_ordered = [v_blue.get(fname, 0.0) for fname in feature_names]
v_red_ordered = [v_red.get(fname, 0.0) for fname in feature_names]

X_blue_raw = np.array([v_blue_ordered], dtype=np.float32)
X_red_raw = np.array([v_red_ordered], dtype=np.float32)

# --- FIX: ADD DATA CLEANING STEP FOR ROBUSTNESS ---
# This ensures the prediction pipeline is identical to the training one.
X_blue_raw = np.nan_to_num(X_blue_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_red_raw = np.nan_to_num(X_red_raw, nan=0.0, posinf=0.0, neginf=0.0)
# --------------------------------------------------

# --- SCALE THE LIVE DATA ---
X_blue_scaled = scaler1.transform(X_blue_raw)
X_red_scaled = scaler2.transform(X_red_raw)
# -------------------------

if np.isnan(X_blue_scaled).any() or np.isnan(X_red_scaled).any():
    print("\n--- ERROR: NaN DETECTED IN INPUT VECTORS ---")
    exit()

print("\nPredicting outcome...")
win_prob_blue = model.predict([X_blue_scaled, X_red_scaled])[0][0]
win_prob_red = 1 - win_prob_blue

print("\n--- PREDICTION ---")
print(f"Blue side ({draft['blue_team']['name']}) win probability: {win_prob_blue:.1%}")
print(f"Red side ({draft['red_team']['name']}) win probability: {win_prob_red:.1%}")
print("------------------")