import os
import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "results", "dataset_build")
RESULT_DIR = os.path.join(BASE_DIR, "results", "preprocessing")

os.makedirs(RESULT_DIR, exist_ok=True)

train_input = os.path.join(INPUT_DIR, "train_pairs.csv")
dev_input = os.path.join(INPUT_DIR, "dev_pairs.csv")

train_output = os.path.join(RESULT_DIR, "train_clean.csv")
dev_output = os.path.join(RESULT_DIR, "dev_clean.csv")

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")



def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Remove citations like [1] or [12]
    text = re.sub(r'\[\d+\]', '', text)
    # Simplify multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_file(input_path, output_path):

    df = pd.read_csv(input_path)

    df["claim"] = df["claim"].fillna("").astype(str).apply(clean_text)
    df["evidence"] = df["evidence"].fillna("").astype(str).apply(clean_text)

    df.to_csv(output_path, index=False)

    print("Saved:", output_path)
    print("Samples:", len(df))


def main():

    preprocess_file(train_input, train_output)
    preprocess_file(dev_input, dev_output)


if __name__ == "__main__":
    main()
    