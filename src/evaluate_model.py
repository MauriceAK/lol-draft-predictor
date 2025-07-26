# src/evaluate_model.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score

data = np.load('data/nn_test_split.npz')
X_test = [
    data['blue_team_idx'],
    data['red_team_idx'],
    data['blue_champions_idx'],
    data['red_champions_idx']
]
y_test = data['y']

model = load_model('models/lol_predictor_nn.h5')
y_pred_probs = model.predict(X_test).flatten()
y_pred = (y_pred_probs >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Detailed report:\n", classification_report(y_test, y_pred))
