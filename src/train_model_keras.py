# src/train_model_keras.py
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.optimizers import Adam

data = np.load('data/nn_train_split.npz')

X_train = [
    data['blue_team_idx'],
    data['red_team_idx'],
    data['blue_champions_idx'],
    data['red_champions_idx']
]
y_train = data['y']

# Inputs
blue_team_input = Input(shape=(1,), name='blue_team_idx')
red_team_input = Input(shape=(1,), name='red_team_idx')
blue_champs_input = Input(shape=(5,), name='blue_champions_idx')
red_champs_input = Input(shape=(5,), name='red_champions_idx')

# Embeddings
team_embedding = Embedding(input_dim=500, output_dim=8)
champ_embedding = Embedding(input_dim=2000, output_dim=8)

blue_team_emb = Flatten()(team_embedding(blue_team_input))
red_team_emb = Flatten()(team_embedding(red_team_input))
blue_champs_emb = Flatten()(champ_embedding(blue_champs_input))
red_champs_emb = Flatten()(champ_embedding(red_champs_input))

x = Concatenate()([blue_team_emb, red_team_emb, blue_champs_emb, red_champs_emb])
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(
    inputs=[blue_team_input, red_team_input, blue_champs_input, red_champs_input],
    outputs=output
)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

model.save('models/lol_predictor_nn.h5')
