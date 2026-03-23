# Claim–Evidence Alignment using SciBERT

## Overview

This project focuses on identifying whether a given evidence sentence supports a scientific claim using Natural Language Processing (NLP). The task is formulated as a binary classification problem, where each claim–evidence pair is classified as either **supported** or **not supported**.

## Approach

The system uses **SciBERT**, a transformer-based language model trained on scientific text. The model is fine-tuned to understand the semantic relationship between a claim and its corresponding evidence sentence.

The overall workflow includes:

* Dataset construction
* Text preprocessing
* Tokenization using SciBERT tokenizer
* Model training using sequence classification
* Evaluation using standard metrics

## Dataset

The dataset consists of structured claim–evidence pairs.

Each record contains:

* **Claim** – the statement to be verified
* **Evidence** – supporting or non-supporting sentence
* **Label** –

  * `1` → Evidence supports the claim
  * `0` → Evidence does not support the claim

### Note

Preprocessed dataset files used for training and evaluation are located in:

```
results/preprocessing/
```

## Model

* Model used: `allenai/scibert_scivocab_uncased`
* Task: Binary classification (Support / Not Support)

The model is fine-tuned on claim–evidence pairs to learn contextual relationships.

## Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score

Visualizations include:

* Confusion Matrix
* Precision–Recall Curve

## Results

The model achieves approximately **70–75% accuracy**, indicating reasonable performance in identifying claim–evidence relationships in scientific text.

## Project Structure

```
Claim_Evidence_Detection_NLP/
│
├── results/
│   ├── preprocessing/        # Preprocessed dataset files
│   ├── dataset_build/       # Intermediate dataset construction files
│   └── evaluation/          # Evaluation outputs and graphs
│
├── src/                     # Source code (training, preprocessing, evaluation)
│
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

* The repository includes preprocessed data for demonstration purposes.
* Model weights are not included due to size constraints.
* The project presents a basic implementation of SciBERT for claim–evidence classification.

## Applications

* Scientific fact verification
* Research paper analysis
* Evidence detection systems
* Information validation

## References

* Vaswani et al., 2017 – Attention is All You Need
* Devlin et al., 2019 – BERT: Pre-training of Deep Bidirectional Transformers
* Beltagy et al., 2019 – SciBERT: A Pretrained Language Model for Scientific Text

---
