import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)


# ==========================
# Load preprocessed data
# ==========================

X_train = np.load(
    "data/X_train.npy"
)

X_val = np.load(
    "data/X_val.npy"
)

X_test = np.load(
    "data/X_test.npy"
)


y_train = np.load(
    "data/y_train.npy"
)

y_val = np.load(
    "data/y_val.npy"
)

y_test = np.load(
    "data/y_test.npy"
)


print("Input shape:", X_train.shape)


# ==========================
# Build RNN Model
# ==========================

model = Sequential([

    tf.keras.Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),


    LSTM(
        128,
        return_sequences=True
    ),

    BatchNormalization(),

    Dropout(0.3),


    LSTM(
        64
    ),

    BatchNormalization(),

    Dropout(0.3),


    Dense(
        32,
        activation="relu"
    ),


    Dropout(0.3),


    Dense(
        1,
        activation="sigmoid"
    )

])


model.summary()


# ==========================
# Compile
# ==========================

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]

)



# ==========================
# Callbacks
# ==========================

early_stop = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


checkpoint = ModelCheckpoint(

    "models/best_unsw_model.keras",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1

)



# ==========================
# Training
# ==========================

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=30,

    batch_size=32,

    callbacks=[
        early_stop,
        checkpoint
    ]

)



# ==========================
# Evaluation
# ==========================

best_model = load_model(
    "models/best_unsw_model.keras"
)


loss, accuracy = best_model.evaluate(
    X_test,
    y_test
)


print("\nTest Loss:", loss)

print(
    "Test Accuracy:",
    accuracy
)



# ==========================
# Save model
# ==========================

best_model.save(
    "models/unsw_model.keras"
)


# Save training history

with open(
    "results/UNSW-NB15_results/unsw_history.pkl",
    "wb"
) as f:

    pickle.dump(
        history.history,
        f
    )


print(
    "\nUNSW model saved successfully."
)
