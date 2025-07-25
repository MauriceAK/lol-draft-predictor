# src/test_match.py

from predict import load_latest_model, predict_winner
import re

def analyze_betting_value(model_output_string, blue_odds, red_odds):
    """
    Parses the model's output string to extract probabilities and compares them
    to the implied probabilities from the given decimal odds.
    """
    try:
        # Use regular expressions to find the single percentage value
        match = re.search(r'(\d+\.\d{2})%', model_output_string)
        if not match:
            raise ValueError("Could not find a probability percentage in the output string.")
            
        probability = float(match.group(1)) / 100
        
        # Determine which team this probability belongs to
        if "Winner: Blue" in model_output_string:
            model_prob_blue = probability
            model_prob_red = 1 - probability
        elif "Winner: Red" in model_output_string:
            model_prob_red = probability
            model_prob_blue = 1 - probability
        else:
            raise ValueError("Could not determine the winning team from the output string.")

    except (ValueError, TypeError) as e:
        print(f"\nCould not parse model probabilities from output: {e}")
        return

    # --- Calculate Implied Probability from Decimal Odds ---
    # Implied Probability = 1 / Decimal Odds
    implied_prob_blue = 1 / blue_odds
    implied_prob_red = 1 / red_odds

    print("\n--- Expected Value Analysis ---")
    
    # --- Blue Team Analysis ---
    blue_edge = model_prob_blue - implied_prob_blue
    print(f"Blue Team (T1):")
    print(f"  Model Probability: {model_prob_blue:.2%}")
    print(f"  Odds Implied Probability: {implied_prob_blue:.2%}")
    if blue_edge > 0:
        print(f"  Result: ✅ BET ON BLUE. Edge: {blue_edge:+.2%}")
    else:
        print(f"  Result: ❌ DO NOT BET ON BLUE. Edge: {blue_edge:+.2%}")
        
    # --- Red Team Analysis ---
    red_edge = model_prob_red - implied_prob_red
    print(f"\nRed Team (Gen.G):")
    print(f"  Model Probability: {model_prob_red:.2%}")
    print(f"  Odds Implied Probability: {implied_prob_red:.2%}")
    if red_edge > 0:
        print(f"  Result: ✅ BET ON RED. Edge: {red_edge:+.2%}")
    else:
        print(f"  Result: ❌ DO NOT BET ON RED. Edge: {red_edge:+.2%}")


def create_t1_vs_geng_test_case():
    """
    Creates a dictionary representing the one-hot encoded features
    for a hypothetical T1 vs. Gen.G match.
    
    Modify the champion picks and bans below to match the actual draft.
    """
    
    # --- Game Context ---
    game_features = {
        'league_LCK': 1,
        'blue_team_T1': 1,
        'red_team_Gen.G': 1,
        'patch_14.14': 1, 
    }
    
    # --- Blue Team (T1) Draft ---
    blue_picks = ['Rakan', 'Xayah', 'Azir', 'Sejuani', 'Jayce']
    blue_bans = ['Hwei', 'Aurora', 'Senna', 'Xin Zhao', 'Taliyah']
    
    # --- Red Team (Gen.G) Draft ---
    red_picks = ['K\'Sante', 'Skarner', 'Twisted Fate', 'Lucian', 'Braum']
    red_bans = ['Pantheon', 'Varus', 'Jarvan IV', 'Trundle', 'Nocturne']
    
    # --- Populate the feature dictionary ---
    for champ in blue_picks:
        game_features[f'blue_pick_{champ}'] = 1
    for champ in blue_bans:
        game_features[f'blue_ban_{champ}'] = 1
    for champ in red_picks:
        game_features[f'red_pick_{champ}'] = 1
    for champ in red_bans:
        game_features[f'red_ban_{champ}'] = 1
        
    return game_features

if __name__ == '__main__':
    # --- DEFINE BETTING ODDS HERE ---
    # Enter the decimal odds from your betting website.
    # Example: T1 is the underdog, Gen.G is the favorite.
    t1_blue_odds = 2.10
    geng_red_odds = 1.65
    
    # 1. Create the dictionary for the T1 vs. Gen.G game
    game_to_predict = create_t1_vs_geng_test_case()
    
    # 2. Load the latest trained model
    predictor_model = load_latest_model()
    
    # 3. Get the prediction and analyze the bet
    if predictor_model:
        result_string = predict_winner(predictor_model, game_to_predict)
        print("\n--- Model Prediction Output ---")
        print(result_string)
        
        # Analyze the betting value based on the model's output and your hard-coded odds
        analyze_betting_value(result_string, t1_blue_odds, geng_red_odds)
