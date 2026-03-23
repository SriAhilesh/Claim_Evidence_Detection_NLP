import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")

RESULT_DIR = os.path.join(BASE_DIR, "results", "dataset_build")
os.makedirs(RESULT_DIR, exist_ok=True)


def load_corpus():

    corpus = {}

    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            paper = json.loads(line)
            corpus[str(paper["doc_id"])] = paper["abstract"]

    return corpus


def build_pairs(input_file, output_file, corpus):

    dataset = []

    with open(input_file, "r", encoding="utf-8") as f:

        for line in f:

            item = json.loads(line)

            claim = item["claim"]
            evidence = item["evidence"]

            if not evidence:

                dataset.append({
                    "claim": claim,
                    "evidence": "",
                    "label": 0
                })

            else:

                for doc_id, ev_list in evidence.items():

                    if doc_id not in corpus:
                        continue

                    sentences = corpus[doc_id]

                    for ev in ev_list:

                        label = ev["label"]

                        for sid in ev["sentences"]:

                            if sid < len(sentences):

                                evidence_text = sentences[sid]

                                y = 1 if label == "SUPPORT" else 0

                                dataset.append({
                                    "claim": claim,
                                    "evidence": evidence_text,
                                    "label": y
                                })

    df = pd.DataFrame(dataset)
    df.to_csv(output_file, index=False)

    print("Saved:", output_file)
    print("Samples:", len(df))


def main():

    corpus = load_corpus()

    train_input = os.path.join(DATA_DIR, "claims_train.jsonl")
    dev_input = os.path.join(DATA_DIR, "claims_dev.jsonl")

    train_output = os.path.join(RESULT_DIR, "train_pairs.csv")
    dev_output = os.path.join(RESULT_DIR, "dev_pairs.csv")

    build_pairs(train_input, train_output, corpus)
    build_pairs(dev_input, dev_output, corpus)

if __name__ == "__main__":
    main()