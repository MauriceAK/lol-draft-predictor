# lolpredictor/model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Subtract, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_siamese_model(input_shape):
    """
    Creates a two-input, shared-weight model for comparing team vectors.
    """
    # 1. Define the "tower" - a sub-network that processes one team vector.
    # These layers have shared weights, meaning we learn a single, robust
    # function for evaluating a team composition's strength.
    tower_input = Input(shape=input_shape, name="tower_input")
    x = Dense(128, activation='relu')(tower_input)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    tower_output = Dense(32, activation='relu')(x) # The final "embedding" for the team

    # The tower model encapsulates this logic
    tower = Model(tower_input, tower_output, name="shared_tower")

    # 2. Define the two inputs for the full siamese model
    input_team1 = Input(shape=input_shape, name="input_team1")
    input_team2 = Input(shape=input_shape, name="input_team2")

    # 3. Process both inputs through the *same* shared tower
    processed_team1 = tower(input_team1)
    processed_team2 = tower(input_team2)

    # 4. Combine the outputs for a final prediction.
    # Subtracting them is a powerful way to explicitly model the difference.
    combined = Subtract()([processed_team1, processed_team2])

    # 5. A final classification head to determine the winner
    x = Dense(16, activation='relu')(combined)
    output = Dense(1, activation='sigmoid', name="win_probability")(x)

    # 6. Build and compile the final model
    model = Model(inputs=[input_team1, input_team2], outputs=output, name="lol_predictor")

        # --- STABILIZED OPTIMIZER ---
    # 1. Lower the learning rate from the default to take smaller steps.
    # 2. Add clipvalue=1.0 for gradient clipping to prevent updates from exploding.
    stable_optimizer = Adam(learning_rate=0.0001, clipvalue=1.0)
    # --------------------------

    model.compile(optimizer=stable_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model