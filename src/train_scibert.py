import os
import pandas as pd
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


###############################################
# PATHS
###############################################

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "results", "preprocessing", "train_clean.csv")
DEV_PATH = os.path.join(BASE_DIR, "results", "preprocessing", "dev_clean.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models", "scibert_model")

os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = "allenai/scibert_scivocab_uncased"


###############################################
# DEBUG MODE (for quick experiments)
###############################################

DEBUG = False


###############################################
# LOAD DATA
###############################################

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    dev_df = pd.read_csv(DEV_PATH)

    train_df["claim"] = train_df["claim"].fillna("").astype(str)
    train_df["evidence"] = train_df["evidence"].fillna("").astype(str)

    dev_df["claim"] = dev_df["claim"].fillna("").astype(str)
    dev_df["evidence"] = dev_df["evidence"].fillna("").astype(str)

    # Keep all evidence strings regardless of length
    # train_df = train_df[train_df["evidence"].str.len() > 5]
    # dev_df = dev_df[dev_df["evidence"].str.len() > 5]

    if DEBUG:
        # Don't subset down so far that we can't learn, even when debugging
        train_df = train_df.sample(min(300, len(train_df)))
        dev_df = dev_df.sample(min(100, len(dev_df)))

    train_dataset = Dataset.from_pandas(train_df[["claim","evidence","label"]])
    dev_dataset = Dataset.from_pandas(dev_df[["claim","evidence","label"]])

    return train_dataset, dev_dataset


###############################################
# TOKENIZER
###############################################

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):

    return tokenizer(
        example["claim"],
        example["evidence"],
        truncation=True,
        padding=False,
        max_length=256
    )


###############################################
# DATASET
###############################################

train_dataset, dev_dataset = load_data()

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset = dev_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","label"]
)

dev_dataset.set_format(
    type="torch",
    columns=["input_ids","attention_mask","label"]
)


###############################################
# MODEL
###############################################

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)


###############################################
# METRICS
###############################################

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    # Changed to macro average to get a more balanced metric across both classes
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


###############################################
# TRAINING SETTINGS
###############################################

training_args = TrainingArguments(

    output_dir=MODEL_DIR,

    learning_rate=3e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,

    num_train_epochs=15,

    weight_decay=0.01,

    lr_scheduler_type="cosine",

    warmup_ratio=0.1,

    evaluation_strategy="steps",
    eval_steps=100,

    save_strategy="steps",
    save_steps=100,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="accuracy",

    greater_is_better=True,

    logging_steps=50,

    fp16=torch.cuda.is_available(),

    report_to="none"
)


###############################################
# TRAINER
###############################################

data_collator = DataCollatorWithPadding(tokenizer)

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=dev_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)


###############################################
# TRAIN
###############################################

trainer.train()

trainer.save_model(MODEL_DIR)

print("Training finished")
print("Model saved to:", MODEL_DIR)