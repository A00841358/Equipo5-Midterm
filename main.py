"""
main.py
Input: a list of comments. Output: a 0/1 label per comment, and the list of negatives (1).

Usage:
    python main.py            # trains if there is no saved model, then classifies
    python main.py --retrain  # force retraining
    python main.py --data data/train.csv
"""
import argparse
import os

from classify import train, classify, get_negatives, load_csv, save_model, load_model

MODEL_PATH = "model.json"


def get_model(data_path, retrain=False):
    if os.path.exists(MODEL_PATH) and not retrain:
        return load_model(MODEL_PATH)
    print(f"Training with {data_path} ...")
    comments, labels = load_csv(data_path)
    model = train(comments, labels)
    save_model(model, MODEL_PATH)
    return model


def run(comments, model):
    labels = classify(comments, model)
    for comment, label in zip(comments, labels):
        print(f"{label}  |  {comment}")
    return get_negatives(comments, model)  # <- this is what the next stage (rewriting) receives


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train_sample.csv")
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()

    model = get_model(args.data, args.retrain)

    comments = [
        "you are an idiot",
        "this is not bad at all, I really liked it",
        "I hate you, you are disgusting",
        "thanks for sharing, beautiful video",
        "what a horrible person you are",
        "I don't think your idea is bad",
    ]

    print("\nLabels (1 = negative, 0 = not negative):")
    negatives = run(comments, model)

    print("\nNegative comments passed to the next stage:")
    for n in negatives:
        print(" -", n)