# Claim–Evidence Alignment using SciBERT

## Overview

This project focuses on determining whether a given evidence sentence supports a scientific claim using Natural Language Processing (NLP). The system is designed as a binary classification task, where each claim–evidence pair is classified as either **supported** or **not supported**.

## Approach

The model is based on **SciBERT**, a transformer-based language model trained on scientific text. The claim and evidence sentences are processed together and passed into the model, which learns to identify semantic relationships between them.

The workflow includes:

* Dataset preparation (claim–evidence pairs)
* Text preprocessing
* Tokenization using SciBERT tokenizer
* Model training using sequence classification
* Evaluation using standard metrics

## Dataset

The dataset consists of structured claim–evidence pairs extracted from scientific text sources.

Each entry contains:

* **Claim** – statement to be verified
* **Evidence** – supporting or non-supporting sentence
* **Label** –

  * `1` → Evidence supports the claim
  * `0` → Evidence does not support the claim

The dataset is divided into:

* Training set
* Development (evaluation) set

## Model

* Model used: `allenai/scibert_scivocab_uncased`
* Task: Binary classification (Support / Not Support)

The model is fine-tuned on the dataset to learn relationships between claims and evidence.

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Visualization techniques such as:

* Confusion Matrix
* Precision–Recall Curve

are used to analyze performance.

## Results

The model achieves approximately **70–75% accuracy**, demonstrating its ability to identify relationships between claims and evidence in scientific text.

## Project Structure

```
Claim_Evidence_Detection_NLP/
│
├── data/        # Sample dataset files
├── results/     # Output visualizations
├── src/         # Training and evaluation scripts
├── README.md
└── requirements.txt
```

## Installation

Install dependencies using:

```
pip install -r requirements.txt
```

## Usage

### Train the model

```
python src/train_scibert.py
```

### Evaluate the model

```
python src/evaluate_scibert.py
```

## Notes

* Model weights are not included due to size constraints.
* The project demonstrates a basic implementation of SciBERT for claim–evidence classification.

## Applications

* Scientific fact verification
* Research paper analysis
* Automated evidence detection
* Information validation systems

## References

* Vaswani et al., 2017 – Attention is All You Need
* Devlin et al., 2019 – BERT: Pre-training of Deep Bidirectional Transformers
* Beltagy et al., 2019 – SciBERT: A Pretrained Language Model for Scientific Text

---
