import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import (
	Conv2D,MaxPooling2D,
	Flatten,Dense,Dropout,BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
# Import preprocessing
from preprocessing.cifar10_preprocessing import load_datasets

###load
train_dataset, validation_dataset, test_dataset, class_names = load_datasets()

###CNN Construction
model = Sequential([
    tf.keras.Input(shape=(32,32,3)),

    Conv2D(32,(3,3),activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64,(3,3),activation="relu"),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128,(3,3),activation="relu"),
    BatchNormalization(),

    Flatten(),

    Dense(256,activation="relu"),
    Dropout(0.5),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(10,activation="softmax")
])

###compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

###Earlystopping
early_stop = EarlyStopping(
	monitor="val_loss",patience=3,restore_best_weights=True
)
###Check point
checkpoint = ModelCheckpoint(
    "models/best_cifar10_model.keras",monitor="val_accuracy",
    save_best_only=True,verbose=1
)

###train
history = model.fit(
    train_dataset,validation_data=validation_dataset,
    epochs=30,callbacks=[early_stop, checkpoint]
)

###Evaluate
best_model = load_model("models/best_cifar10_model.keras")

loss, accuracy = best_model.evaluate(test_dataset)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

###save
best_model.save("models/cifar10_model.keras")
print("Model saved successfully.")
with open("results/CIFAR10_results/cifar10_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
print("\nTraining completed successfully.")
print("Model saved in models/cifar10_model.keras")
