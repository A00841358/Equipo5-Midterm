"""
download_hf.py
Downloads an English dataset from Hugging Face and saves it as CSV (text,label) for classify.py.

Requires:  pip install datasets
Usage:     python download_hf.py

Dataset: tweet_eval / offensive  (English tweets, label: 1 = offensive, 0 = not offensive).
Check the dataset card on Hugging Face for license and details.
"""
import csv
import os

from datasets import load_dataset

DATASET, CONFIG = "cardiffnlp/tweet_eval", "offensive"
TEXT_COL, LABEL_COL = "text", "label"
MAX_ROWS = 20000

os.makedirs("data", exist_ok=True)
ds = load_dataset(DATASET, CONFIG)

with open("data/train.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    for row in list(ds["train"])[:MAX_ROWS]:
        w.writerow([row[TEXT_COL].replace("\n", " "), int(row[LABEL_COL])])

with open("data/test.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    for row in ds["test"]:
        w.writerow([row[TEXT_COL].replace("\n", " "), int(row[LABEL_COL])])

print("Done: data/train.csv and data/test.csv")