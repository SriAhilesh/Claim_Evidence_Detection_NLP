import os
import pandas as pd
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef
)

import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEV_PATH = os.path.join(BASE_DIR, "results", "preprocessing", "dev_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "scibert_model")

RESULT_DIR = os.path.join(BASE_DIR, "results", "evaluation")
os.makedirs(RESULT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------

df = pd.read_csv(DEV_PATH)
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()

# -----------------------------
# Prediction
# -----------------------------

predictions = []
probabilities = []

true_labels = df["label"].tolist()

for idx, row in df.iterrows():

    inputs = tokenizer(
        str(row["claim"]),
        str(row["evidence"]),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()

    predictions.append(pred)
    probabilities.append(probs[:,1].item())

# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
roc_auc = roc_auc_score(true_labels, probabilities)
mcc = matthews_corrcoef(true_labels, predictions)

report = classification_report(true_labels, predictions)

# Save classification report

with open(os.path.join(RESULT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Save metrics summary

metrics_text = f"""
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1 Score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}
MCC: {mcc:.4f}
"""

with open(os.path.join(RESULT_DIR, "metrics_summary.txt"), "w") as f:
    f.write(metrics_text)

print(metrics_text)
print(report)

# -----------------------------
# Confusion Matrix
# -----------------------------

cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Misaligned","Aligned"],
    yticklabels=["Misaligned","Aligned"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.close()

# -----------------------------
# ROC Curve
# -----------------------------

fpr, tpr, _ = roc_curve(true_labels, probabilities)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1], [0,1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig(os.path.join(RESULT_DIR, "roc_curve.png"))
plt.close()

# -----------------------------
# Precision Recall Curve
# -----------------------------

precision_vals, recall_vals, _ = precision_recall_curve(true_labels, probabilities)

plt.figure()

plt.plot(recall_vals, precision_vals)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig(os.path.join(RESULT_DIR, "precision_recall_curve.png"))
plt.close()

# -----------------------------
# Probability Distribution
# -----------------------------

plt.figure()

sns.histplot(probabilities, bins=20)

plt.xlabel("Prediction Probability (Aligned)")
plt.title("Prediction Confidence Distribution")

plt.savefig(os.path.join(RESULT_DIR, "prediction_probability_distribution.png"))
plt.close()

print("All evaluation results saved to:")
print(RESULT_DIR)