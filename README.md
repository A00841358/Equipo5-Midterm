# Equipo5-Midterm

## Setup

python3 -m venv venv
source venv/bin/activate       
Windows: venv\Scripts\activate
pip install -r requirements.txt   # only needed to download the dataset

## Get the data and train the model

python3 hugging_face_download.py:
creates data/train.csv and data/test.csv

python3 main.py --data data/train.csv --retrain:
trains and saves model.json

## Evaluate and compare
python3 compare_knn.py     # logistic regression vs. KNN baseline