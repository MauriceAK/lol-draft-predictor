import joblib
import pandas as pd
import os

def load_latest_model(model_dir='models'):
    """
    Loads the most recently saved model from the specified directory.
    """
    try:
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
        latest_model_file = sorted(model_files, reverse=True)[0]
        model_path = os.path.join(model_dir, latest_model_file)
        model = joblib.load(model_path)
        print(f"Loaded latest model: {latest_model_file}")
        return model
    except (FileNotFoundError, IndexError):
        print(f"Error: No model found in the '{model_dir}' directory. Please train a model first.")
        return None

def predict_winner(model, game_data_dict):
    """
    Predicts the winner for a single game provided as a dictionary.
    
    Args:
        model: The trained model object.
        game_data_dict: A dictionary where keys are feature names and values are 1 or 0.
    
    Returns:
        A string indicating the predicted winner ('Blue' or 'Red').
    """
    if model is None:
        return "Model not available."
        
    # Convert the dictionary to a pandas DataFrame
    df = pd.DataFrame([game_data_dict])
    
    # Ensure all model features are present, filling missing ones with 0
    # This handles cases where a new game might not have all one-hot encoded features
    model_features = model.get_booster().feature_names
    for col in model_features:
        if col not in df.columns:
            df[col] = 0
            
    # Reorder columns to match the model's training order
    df = df[model_features]

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    
    if prediction == 1:
        return f"Predicted Winner: Blue (Probability: {probability[1]:.2%})"
    else:
        return f"Predicted Winner: Red (Probability: {probability[0]:.2%})"


if __name__ == '__main__':
    # Example of how to use the prediction script
    # This dictionary represents one game's draft features.
    # In a real application, you would generate this from live game data.
    example_game = {
        'league_LCK': 1,
        'blue_team_T1': 1,
        'red_team_Gen.G': 1,
        'patch_13.14': 1,
        'blue_pick_Aatrox': 1,
        'red_pick_Fiora': 1
        # ... and all other features would be 0 for this game
    }
    
    predictor_model = load_latest_model()
    result = predict_winner(predictor_model, example_game)
    print(result)