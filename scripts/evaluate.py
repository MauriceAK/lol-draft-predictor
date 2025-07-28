# scripts/evaluate.py

import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from lolpredictor.etl import load_and_prep_data
import config
import seaborn as sns
import matplotlib.pyplot as plt

print("Loading model and preparing test data...")

# 1. Load the trained model and feature maps
MODEL_PATH = f"{config.MODELS_DIR}/01_siamese_model.keras"
MAPS_PATH = f"{config.MODELS_DIR}/feature_maps.pkl"
SCALER1_PATH = f"{config.MODELS_DIR}/scaler1.pkl"
SCALER2_PATH = f"{config.MODELS_DIR}/scaler2.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(MAPS_PATH, 'rb') as f:
    maps = pickle.load(f)
with open(SCALER1_PATH, 'rb') as f: scaler1 = pickle.load(f)
with open(SCALER2_PATH, 'rb') as f: scaler2 = pickle.load(f)

# 2. Load the full dataset using the same ETL process as training
match_df, _ = load_and_prep_data()
assert not match_df.empty, "No data loaded. Cannot evaluate."


# 3. Prepare the data exactly as in the training script
X1 = np.array([list(d.values()) for d in match_df['team1_vector']], dtype=np.float32)
X2 = np.array([list(d.values()) for d in match_df['team2_vector']], dtype=np.float32)
y = match_df['label'].values

# 4. Split the data to get the *exact same test set* used during training
# Using the same random_state (42) is crucial here.
_, X1_test, _, X2_test, _, y_test = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42, stratify=y
)

# --- SCALE THE TEST DATA ---
X1_test_scaled = scaler1.transform(X1_test)
X2_test_scaled = scaler2.transform(X2_test)
# -------------------------

print(f"Evaluating model on {len(y_test)} test samples...")

# 5. Make predictions on the test set
y_pred_proba = model.predict([X1_test_scaled, X2_test_scaled])
y_pred_class = (y_pred_proba > 0.5).astype(int)

# 6. Generate and print evaluation metrics
print("\n--- MODEL EVALUATION REPORT ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_class):.2%}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_class, target_names=['Red Team Win', 'Blue Team Win']))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred_class)
print(cm)
print("\nTrue Neg (Red Wins Correct):", cm[0][0])
print("False Pos (Red Wins as Blue):", cm[0][1])
print("False Neg (Blue Wins as Red):", cm[1][0])
print("True Pos (Blue Wins Correct):", cm[1][1])
print("-----------------------------\n")

# Optional: Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Red Win', 'Blue Win'], yticklabels=['Red Win', 'Blue Win'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('evaluation_confusion_matrix.png')
print("Confusion matrix plot saved to evaluation_confusion_matrix.png")