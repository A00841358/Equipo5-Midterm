
import argparse
import time
import tracemalloc

from classify import train, classify, load_csv
from knn import knn_train, knn_classify

parser = argparse.ArgumentParser()
parser.add_argument("--train", default="data/train.csv")
parser.add_argument("--test", default="data/test.csv")
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

train_comments, train_labels = load_csv(args.train)
test_comments, test_labels = load_csv(args.test)
k = args.k


def quality(predictions, labels):
    true_positives = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    false_positives = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    false_negatives = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)
    accuracy = sum(1 for p, y in zip(predictions, labels) if p == y) / len(labels)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, f1


def peak_memory_mb(function):
    tracemalloc.start()
    function()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes / 1e6


print(f"Training data: {len(train_comments)} comments | test data: {len(test_comments)} comments | KNN k = {k}")
print("Both algorithms use the same n-gram preprocessing.\n")

start = time.perf_counter()
logistic_model = train(train_comments, train_labels)
logistic_train_seconds = time.perf_counter() - start
logistic_memory = peak_memory_mb(lambda: train(train_comments, train_labels, epochs=1))

start = time.perf_counter()
knn_model = knn_train(train_comments, train_labels)
knn_train_seconds = time.perf_counter() - start
knn_memory = peak_memory_mb(lambda: knn_train(train_comments, train_labels))

start = time.perf_counter()
logistic_predictions = classify(test_comments, logistic_model)
logistic_classify_seconds = time.perf_counter() - start

print("Classifying with KNN")
start = time.perf_counter()
knn_predictions = knn_classify(test_comments, knn_model, k)
knn_classify_seconds = time.perf_counter() - start

logistic_quality = quality(logistic_predictions, test_labels)
knn_quality = quality(knn_predictions, test_labels)
always_zero_accuracy = sum(1 for y in test_labels if y == 0) / len(test_labels)



print("\nClassification time per comment as the training data grows")
print(f"{'training comments':>17} | {'logistic regression':>19} | {'KNN':>10}")
print("-" * 55)
sample = test_comments[:100]
sizes = [n for n in (1000, 2000, 4000, 8000) if n < len(train_comments)] + [len(train_comments)]
for size in sizes:
    small_logistic = train(train_comments[:size], train_labels[:size], epochs=1)
    small_knn = knn_train(train_comments[:size], train_labels[:size])

    start = time.perf_counter()
    classify(sample, small_logistic)
    logistic_time = (time.perf_counter() - start) / len(sample) * 1e6

    start = time.perf_counter()
    knn_classify(sample, small_knn, k)
    knn_time = (time.perf_counter() - start) / len(sample) * 1e6

    print(f"{size:>17} | {logistic_time:>16.0f} us | {knn_time:>7.0f} us")