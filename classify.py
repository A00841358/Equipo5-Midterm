"""
Logistic regression with n-grams.

Labels: 1 = negative, 0 = not negative

Negation word list from NEGATE list of VADER (Hutto & Gilbert, 2014):
  Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
  Sentiment Analysis of Social Media Text. Eighth International Conference on
  Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

"""
import csv
import json
import math
import re


VADER_NEGATE = {
    "aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
    "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
    "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
    "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
    "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing",
    "nowhere", "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
    "oughtn't", "shan't", "shouldn't", "wasn't", "weren't", "without",
    "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite",
}

#Extras
EXTRA_NEGATORS = {"no", "nobody", "hardly", "barely"}
NEGATORS = VADER_NEGATE | EXTRA_NEGATORS

PUNCTUATION = {".", ",", ";", ":", "!", "?"}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[.,;:!?]")


NEGATION_WINDOW = 3


# Aquí se procesan los comentarios para que el modelo los pueda leer justamente.
#La negación se marca con un prefijo "NOT_" a las palabras que estén dentro de la ventana de negación.
#Se cuenta con una ventana de 3 palabras después de un negador para marcar la negación.
#De esta manera, el modelo puede aprender que "not good" es negativo, mientras que "good" es positivo.
def tokenize(text):
    text = text.lower().replace("’", "'")
    return TOKEN_PATTERN.findall(text)


def mark_negation(tokens):
    marked_tokens = []
    words_left_to_mark = 0
    for token in tokens:
        if token in PUNCTUATION:
            words_left_to_mark = 0
            marked_tokens.append(token)
        elif token in NEGATORS:
            words_left_to_mark = NEGATION_WINDOW
            marked_tokens.append(token)
        elif words_left_to_mark > 0:
            marked_tokens.append("NOT_" + token)
            words_left_to_mark -= 1
        else:
            marked_tokens.append(token)
    return marked_tokens


def extract_ngrams(text, max_ngram_size=2):
    words = [token for token in mark_negation(tokenize(text)) if token not in PUNCTUATION]
    ngrams = []
    for size in range(1, max_ngram_size + 1):
        for start in range(len(words) - size + 1):
            ngrams.append("_".join(words[start:start + size]))
    return ngrams


#Construccion del modelo de regresión logística con n-gramas.
#Se utiliza la función sigmoide para mapear la salida a un rango entre 0 y 1, lo que permite interpretar la salida como una probabilidad de que el comentario sea negativo.
#Se construye un vocabulario de n-gramas a partir de los comentarios de entrenamiento, y se entrena el modelo utilizando descenso de gradiente para ajustar los pesos de cada n-grama y el sesgo del modelo.

def sigmoid(score):
    return 1 / (1 + math.exp(-score))


def build_vocabulary(comments, min_count=1):
    counts = {}
    for comment in comments:
        for ngram in extract_ngrams(comment):
            counts[ngram] = counts.get(ngram, 0) + 1
    return {ngram for ngram, count in counts.items() if count >= min_count}


def predict(comment, model):
    score = model["bias"]
    for ngram in extract_ngrams(comment):
        score += model["weights"].get(ngram, 0.0)
    return sigmoid(score)


#Entrenamiento, ve cuanto fue de diferencia entre la predicción y la etiqueta real, y ajusta los pesos de los n-gramas de acuerdo a cuanto fue la perdida.

def train(comments, labels, epochs=10, learning_rate=0.1, min_count=1, l2=0.0):
    weights = {ngram: 0.0 for ngram in build_vocabulary(comments, min_count)}
    model = {"weights": weights, "bias": 0.0}

    for epoch in range(1, epochs + 1):
        for comment, label in zip(comments, labels):
            probability = predict(comment, model)
            error = probability - label
            for ngram in extract_ngrams(comment):
                if ngram in weights:
                    weights[ngram] -= learning_rate * (error + l2 * weights[ngram])
            model["bias"] -= learning_rate * error

    return model


def classify(comments, model, threshold=0.5):
    return [1 if predict(comment, model) >= threshold else 0 for comment in comments]


def get_negatives(comments, model):
    labels = classify(comments, model)
    return [comment for comment, label in zip(comments, labels) if label == 1]


def load_csv(path):
    """CSV with columns: text,label  (label: 1 = negative, 0 = not negative)"""
    comments, labels = [], []
    with open(path, encoding="utf-8", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            comments.append(row["text"])
            labels.append(int(row["label"]))
    return comments, labels


def save_model(model, path):
    with open(path, "w", encoding="utf-8") as model_file:
        json.dump(model, model_file)


def load_model(path):
    with open(path, encoding="utf-8") as model_file:
        return json.load(model_file)