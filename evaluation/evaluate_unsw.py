import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================
# Load trained RNN model
# ==========================

model = load_model(
    "models/unsw_model.keras"
)


# ==========================
# Load test data
# ==========================

X_test = np.load(
    "data/X_test.npy"
)

y_test = np.load(
    "data/y_test.npy"
)


print(
    "Test shape:",
    X_test.shape
)


# ==========================
# Prediction
# ==========================

y_prob = model.predict(
    X_test
)


# Convert probability to class

y_pred = (
    y_prob >= 0.5
).astype(int)



# ==========================
# Metrics
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


roc_auc = roc_auc_score(
    y_test,
    y_prob
)



# ==========================
# Classification report
# ==========================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Normal",
        "Attack"
    ]
)


print(report)



# ==========================
# Confusion matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Attack"
    ]
)


disp.plot(
    cmap="Blues"
)


plt.title(
    "UNSW-NB15 Confusion Matrix"
)


plt.savefig(
    "results/UNSW-NB15_results/unsw_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()



# ==========================
# Print results
# ==========================

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC AUC  :", roc_auc)



# ==========================
# Save results
# ==========================

with open(
    "results/UNSW-NB15_results/unsw_metrics.txt",
    "w"
) as f:

    f.write(
        f"Accuracy : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {precision:.4f}\n"
    )

    f.write(
        f"Recall   : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score : {f1:.4f}\n"
    )

    f.write(
        f"ROC AUC  : {roc_auc:.4f}\n"
    )


with open(
    "results/UNSW-NB15_results/unsw_classification_report.txt",
    "w"
) as f:

    f.write(report)



print(
    "\nUNSW evaluation completed successfully."
)
