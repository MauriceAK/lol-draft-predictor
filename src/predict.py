import joblib
import pandas as pd
import os
from data_processing import _precompute_stats # We now import the helper function

def load_latest_model(model_dir='models'):
    """
    Loads the most recently saved model from the specified directory.
    """
    try:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        if not model_files:
            print(f"Error: No model files (.joblib) found in the '{model_dir}' directory.")
            return None
        latest_model_file = sorted(model_files, reverse=True)[0]
        model_path = os.path.join(model_dir, latest_model_file)
        model = joblib.load(model_path)
        print(f"Loaded latest model: {latest_model_file}")
        return model
    except FileNotFoundError:
        print(f"Error: The directory '{model_dir}' was not found.")
        return None

def create_features_for_single_game(new_game_dict, historical_df):
    """
    Creates the full feature vector for a single new game, including statistical features.
    """
    print("Creating features for the new game...")
    
    # Pre-compute stats from the entire historical dataset
    historical_stats = _precompute_stats(historical_df)
    patches = sorted(historical_df['patch'].unique())
    
    current_patch = new_game_dict['patch']
    current_patch_index = patches.index(current_patch) if current_patch in patches else -1
    past_patches = patches[:current_patch_index]
    form_patches = patches[max(0, current_patch_index - 3) : current_patch_index]
    
    game_features = {}

    # Calculate statistical features for the new game
    for side in ['blue', 'red']:
        team = new_game_dict[f'{side}_team']
        wins = sum(historical_stats['team_wins_per_patch'][team].get(p, 0) for p in form_patches)
        games = sum(historical_stats['team_games_per_patch'][team].get(p, 0) for p in form_patches)
        game_features[f'{side}_team_winrate_form'] = wins / games if games > 5 else 0.5

        for i in range(5):
            p, c = new_game_dict[f'{side}_players'][i], new_game_dict[f'{side}_champions'][i]
            # --- FIX: Define the team variable 't' here ---
            t = new_game_dict[f'{side}_team']
            
            p_c_wins = sum(historical_stats['player_champ_wins_per_patch'][p][c].get(p_patch, 0) for p_patch in past_patches)
            p_c_games = sum(historical_stats['player_champ_games_per_patch'][p][c].get(p_patch, 0) for p_patch in past_patches)
            game_features[f'{side}_p{i+1}_champ_wr'] = p_c_wins / p_c_games if p_c_games > 3 else 0.5
            
            p_t_c_wins = sum(historical_stats['player_team_champ_wins_per_patch'][(p, t)][c].get(p_patch, 0) for p_patch in past_patches)
            p_t_c_games = sum(historical_stats['player_team_champ_games_per_patch'][(p, t)][c].get(p_patch, 0) for p_patch in past_patches)
            game_features[f'{side}_p{i+1}_team_champ_wr'] = p_t_c_wins / p_t_c_games if p_t_c_games > 2 else 0.5

            c_p_wins = historical_stats['champ_patch_wins'][c].get(current_patch, 0)
            c_p_games = historical_stats['champ_patch_games'][c].get(current_patch, 0)
            game_features[f'{side}_c{i+1}_patch_wr'] = c_p_wins / c_p_games if c_p_games > 1 else 0.5

    # Add one-hot encoded features
    game_features[f"league_{new_game_dict['league']}"] = 1
    game_features[f"blue_team_{new_game_dict['blue_team']}"] = 1
    game_features[f"red_team_{new_game_dict['red_team']}"] = 1
    for c in new_game_dict['blue_champions']: game_features[f'blue_pick_{c}'] = 1
    for c in new_game_dict['red_champions']: game_features[f'red_pick_{c}'] = 1
    for c in new_game_dict['blue_bans']: game_features[f'blue_ban_{c}'] = 1
    for c in new_game_dict['red_bans']: game_features[f'red_ban_{c}'] = 1
    
    return game_features


def predict_winner(model, new_game_dict, historical_df):
    """
    Predicts the winner for a single new game.
    """
    if model is None: return "Model not available."
    
    # 1. Create the full feature vector for the new game
    game_features = create_features_for_single_game(new_game_dict, historical_df)
    
    # 2. Prepare DataFrame for prediction
    model_features = model.get_booster().feature_names
    prediction_df = pd.DataFrame(columns=model_features, index=[0]).fillna(0)

    for feature, value in game_features.items():
        if feature in prediction_df.columns:
            prediction_df[feature] = value
        else:
            print(f"Warning: Feature '{feature}' from test case not found in model's training features. Ignoring.")

    # 3. Make prediction
    probabilities = model.predict_proba(prediction_df)[0]
    prob_red_wins, prob_blue_wins = probabilities[0], probabilities[1]
    
    output = [
        "--- Betting Thresholds ---",
        f"Blue Team Model Win Probability: {prob_blue_wins:.2%}",
        f"  -> Bet on Blue if website odds imply a probability BELOW {prob_blue_wins:.2%}",
        "",
        f"Red Team Model Win Probability: {prob_red_wins:.2%}",
        f"  -> Bet on Red if website odds imply a probability BELOW {prob_red_wins:.2%}"
    ]
    return "\n".join(output)
