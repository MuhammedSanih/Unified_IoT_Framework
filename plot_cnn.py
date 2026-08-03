import pickle
import matplotlib.pyplot as plt

with open("results/CIFAR10_results/cifar10_history.pkl", "rb") as f:
    history = pickle.load(f)
### CNN Training and validation accuracy
plt.figure(figsize=(8,5))

plt.plot(history["accuracy"], label="Training Accuracy")
plt.plot(history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training and Validation Accuracy")
plt.legend()

plt.grid(True)

plt.savefig("results/CIFAR10_results/cnn_accuracy.png",
            dpi=300,
            bbox_inches="tight")

plt.show()

### CNN Training and validation loss
plt.figure(figsize=(8,5))

plt.plot(history["loss"], label="Training Loss")
plt.plot(history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training and Validation Loss")
plt.legend()

plt.grid(True)

plt.savefig("results/CIFAR10_results/cnn_loss.png",
            dpi=300,
            bbox_inches="tight")

plt.show()
