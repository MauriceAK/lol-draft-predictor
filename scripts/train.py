# scripts/train.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # Import the scaler
from lolpredictor.etl import load_and_prep_data
from lolpredictor.model import create_siamese_model
import config
import pickle
import duckdb

def sanitize_maps(maps):
    clean = {}
    for name, m in maps.items():
        if isinstance(m, dict):
            d = {}
            for k, v in m.items():
              
                # namedtuple → dict
                if hasattr(v, "_asdict"):
                    d[k] = v._asdict()
                else:
                    d[k] = v
            clean[name] = d
   
        # if you ever store DataFrames/Series directly:
        elif isinstance(m, pd.DataFrame):
            clean[name] = m.to_dict(orient="list")
        else:
            clean[name] = m
    return clean

print("Starting data loading and preparation...")
# This single function now returns both the training data and the maps needed for prediction
match_df, feature_maps = load_and_prep_data()
print(f"Loaded {len(match_df)} total matches.")
assert len(match_df) > 0, "ETL process failed: No matches were loaded."

# --- NEW: FILTERING FOR MAJOR LEAGUE DATA ---
print("Filtering for major league matchups (LCS, LCK, LPL, LEC)...")

# Get the raw match data which contains team and opponent IDs
raw_matches_df = pd.DataFrame(feature_maps['raw_matches'])

# Define the major leagues
major_leagues = {'LCS', 'LCK', 'LPL', 'LEC', 'LTA S', 'LTA N', 'CBLOL', 'LCP'}

# Connect to the database to get league information for each team
try:
    con = duckdb.connect(config.DB_PATH, read_only=True)
    # This query gets the primary domestic league for each team
    league_info_df = con.execute("""
        SELECT DISTINCT teamid, league 
        FROM raw_player_stats 
        WHERE league IN ('LCS', 'LCK', 'LPL', 'LEC', 'LTA S', 'LTA N', 'CBLOL', 'LCP')
    """).df()
    con.close()
except Exception as e:
    print(f"Database connection failed: {e}")
    # Exit gracefully if the database isn't available
    exit()

# Map the league to the team and the opponent team in each match
# The raw_matches_df has columns 'teamid' and 'opp_teamid'
raw_matches_df = pd.merge(raw_matches_df, league_info_df.rename(columns={'league': 'team_league'}), on='teamid', how='left')
raw_matches_df = pd.merge(raw_matches_df, league_info_df.rename(columns={'league': 'opp_team_league'}), left_on='opp_teamid', right_on='teamid', how='left')

# Filter for matches where BOTH teams are in a major league
major_league_matches_df = raw_matches_df.dropna(subset=['team_league', 'opp_team_league'])
major_league_match_ids = set(major_league_matches_df['match_id'])

# Apply the filter to our main DataFrame used for training
original_count = len(match_df)
match_df = match_df[match_df['match_id'].isin(major_league_match_ids)].copy()
print(f"Filtered from {original_count} to {len(match_df)} matches for model training.")
assert len(match_df) > 0, "No major league matches found after filtering. Check data and league names."
# --- END OF NEW SECTION ---


# Unpack the feature vectors from the DataFrame into numpy arrays
X1 = np.array([list(d.values()) for d in match_df['team1_vector']])
X2 = np.array([list(d.values()) for d in match_df['team2_vector']])
y = match_df['label'].values

X1 = np.nan_to_num(X1, nan=0.0, posinf=0.0, neginf=0.0)
X2 = np.nan_to_num(X2, nan=0.0, posinf=0.0, neginf=0.0)
print("Data cleaned of any NaN/inf values.")

# Get feature names for later use if needed (from the first vector)
feature_names = list(match_df['team1_vector'].iloc[0].keys())
print(f"Vector size: {len(feature_names)} features.")

# Split data for training and validation
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42, stratify=y
)

# --- FEATURE SCALING ---
# 1. Initialize two scalers, one for each input tower of the siamese network.
scaler1 = StandardScaler()
scaler2 = StandardScaler()

# 2. Fit the scalers *only* on the training data to prevent data leakage.
X1_train_scaled = scaler1.fit_transform(X1_train)
X2_train_scaled = scaler2.fit_transform(X2_train)

# 3. Transform the validation data using the *already fitted* scalers.
X1_val_scaled = scaler1.transform(X1_val)
X2_val_scaled = scaler2.transform(X2_val)
# -----------------------

print(f"Training on {len(X1_train)} samples, validating on {len(X1_val)} samples.")

# Create and train the model
input_shape = (X1_train_scaled.shape[1],)
model = create_siamese_model(input_shape)
model.summary() # Print the model structure

# Use callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, monitor='val_loss', factor=0.2)
]

history = model.fit(
    [X1_train_scaled, X2_train_scaled], # Note the input is a list of two arrays
    y_train,
    validation_data=([X1_val_scaled, X2_val_scaled], y_val),
    epochs=100, # Increased epochs, EarlyStopping will handle the rest
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)

# --- SAVE THE SCALERS ---
# We need to save the fitted scalers to use them for prediction.
with open(f"{config.MODELS_DIR}/scaler1.pkl", 'wb') as f:
    pickle.dump(scaler1, f)
with open(f"{config.MODELS_DIR}/scaler2.pkl", 'wb') as f:
    pickle.dump(scaler2, f)
print("\nScalers saved successfully.")
# ------------------------

# Save the trained model in the recommended Keras format
model_path = f"{config.MODELS_DIR}/01_siamese_model.keras"
model.save(model_path)
print(f"\nModel saved successfully to {model_path}")

# You can also save the feature maps and feature names for the prediction script
import pickle
clean_maps = sanitize_maps(feature_maps)
with open(f"{config.MODELS_DIR}/feature_maps.pkl", 'wb') as f:
    pickle.dump(clean_maps, f)
with open(f"{config.MODELS_DIR}/feature_names.pkl", 'wb') as f:
    pickle.dump(feature_names, f)
print("Feature maps and names saved.")