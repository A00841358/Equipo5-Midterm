
import math
from collections import Counter

from classify import extract_ngrams


def vectorize(comment):
    counts = Counter(extract_ngrams(comment))
    norm = math.sqrt(sum(count * count for count in counts.values()))
    return counts, norm


def knn_train(comments, labels):
    vectors = [vectorize(comment) for comment in comments]
    return {"vectors": vectors, "labels": labels}


def cosine_similarity(counts_a, norm_a, counts_b, norm_b):
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot_product = sum(count * counts_b.get(ngram, 0) for ngram, count in counts_a.items())
    return dot_product / (norm_a * norm_b)


def knn_predict(comment, model, k=5):
    counts, norm = vectorize(comment)
    similarities = []
    for (other_counts, other_norm), label in zip(model["vectors"], model["labels"]):
        similarities.append((cosine_similarity(counts, norm, other_counts, other_norm), label))
    nearest = sorted(similarities, key=lambda pair: pair[0], reverse=True)[:k]
    negative_votes = sum(label for _, label in nearest)
    return 1 if negative_votes > k / 2 else 0


def knn_classify(comments, model, k=5):
    return [knn_predict(comment, model, k) for comment in comments]