import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "scibert_model")
DEV_PATH = os.path.join(BASE_DIR, "results", "preprocessing", "dev_clean.csv")
REPORT_DIR = os.path.join(BASE_DIR, "results", "evaluation")

os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("Loading dev data...")
    dev_df = pd.read_csv(DEV_PATH)
    dev_df["claim"] = dev_df["claim"].fillna("").astype(str)
    dev_df["evidence"] = dev_df["evidence"].fillna("").astype(str)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    # Predict batches
    print("Generating predictions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions = []
    labels = dev_df["label"].tolist()
    
    # Process in batches to handle memory
    batch_size = 32
    for i in range(0, len(dev_df), batch_size):
        batch = dev_df.iloc[i:min(i+batch_size, len(dev_df))]
        
        inputs = tokenizer(
            batch["claim"].tolist(),
            batch["evidence"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            
    print("Evaluating metrics...")
    acc = accuracy_score(labels, predictions)
    report = classification_report(labels, predictions, digits=4)
    
    print("\n" + "="*50)
    print(f"Accuracy: {acc:.4f}")
    print("="*50)
    print(report)
    print("="*50)
    
    report_path = os.path.join(REPORT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
