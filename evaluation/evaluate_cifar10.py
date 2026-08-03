import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,precision_score,recall_score,
    f1_score,classification_report,confusion_matrix,
    ConfusionMatrixDisplay
)
from preprocessing.cifar10_preprocessing import load_datasets

### Load
_, _, test_dataset, class_names = load_datasets()
model = load_model("models/cifar10_model.keras")

### Collect true labels
y_true = []

for images, labels in test_dataset:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)

### Predict
y_prob = model.predict(test_dataset)
y_pred = np.argmax(y_prob, axis=1)

### Calculate Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true,y_pred,average="weighted")
recall = recall_score(y_true,y_pred,average="weighted")
f1 = f1_score(y_true,y_pred,average="weighted")

report = classification_report(y_true,y_pred,target_names=class_names)

### Confusion metrics
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,display_labels=class_names
)
disp.plot(cmap="Blues",xticks_rotation=45)

plt.title("CIFAR-10 Confusion Matrix")
plt.savefig("results/CIFAR10_results/cifar10_confusion_matrix.png")
plt.show()

### Save and Print
print(report)
with open(
	"results/CIFAR10_results/cifar10_classification_report.txt", "w"
	) as f:
    f.write(report)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

with open("results/CIFAR10_results/cifar10_metrics.txt", "w") as f:
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
print("\nCIFAR-10 evaluation completed successfully.")
