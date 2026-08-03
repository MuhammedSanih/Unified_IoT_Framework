import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/unsw_model.keras")

# ==========================
# Load test data
# ==========================
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")
# Load training history
with open("results/UNSW-NB15_results/unsw_history.pkl", "rb") as f:
    history = pickle.load(f)
###RNN training and validation accuracy
plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.title("RNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/UNSW-NB15_results/rnn_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

###RNN Training and validation loss
plt.figure(figsize=(8,5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.title("RNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/UNSW-NB15_results/rnn_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================
# Predict probabilities
# ==========================
y_prob = model.predict(X_test, verbose=0).ravel()

# ==========================
# ROC Curve
# ==========================
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr,
         label=f"LSTM-RNN (AUC = {roc_auc:.4f})",
         linewidth=2)

plt.plot([0,1], [0,1], 'k--', label="Random Guess")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - UNSW-NB15 RNN Model")
plt.legend(loc="lower right")
plt.grid(True)

plt.savefig("results/UNSW-NB15_results/rnn_roc_curve.png", dpi=300)
plt.show()

print(f"ROC AUC = {roc_auc:.4f}")
