import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import datetime

def train_model(data_path, model_output_dir):
    """
    Trains an XGBoost classifier on the processed ML features.
    """
    print("--- Starting Model Training ---")
    
    # --- Load Data ---
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded data from {data_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please run the data pipeline first.")
        return

    # --- Prepare Data for Training ---
    X = df.drop('winner_is_blue', axis=1)
    y = df['winner_is_blue']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

    # --- Train XGBoost Model ---
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    model.fit(X_train, y_train)

    # --- Evaluate Model ---
    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # --- Save Model and Logs ---
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Versioning with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'lol_win_predictor_{timestamp}.joblib'
    model_filepath = os.path.join(model_output_dir, model_filename)
    joblib.dump(model, model_filepath)
    print(f"\nModel saved to {model_filepath}")

    # Save training log
    log_filepath = os.path.join(model_output_dir, 'training_log.txt')
    with open(log_filepath, 'a') as f:
        f.write(f"--- Log Entry: {timestamp} ---\n")
        f.write(f"Model: {model_filename}\n")
        f.write(f"Training Data: {data_path}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(report)
        f.write("\n\n")
    print(f"Training log updated at {log_filepath}")


if __name__ == '__main__':
    # We will train on the main regions data by default
    train_model(data_path='data/processed/main_regions_ml_features.csv', model_output_dir='models')