# =====================================================
# Sentiment Analysis | ML + LSTM
# Best Model Selection using F1 Score
# =====================================================

import os
import re
import time
import warnings
warnings.filterwarnings("ignore")

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import pandas as pd
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# ========================
# NLP SETUP
# ========================
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in STOP_WORDS)
    return " ".join(LEMMATIZER.lemmatize(w) for w in text.split())

# ========================
# LOAD DATA
# ========================
DATA_PATH = r"C:\Users\bmoha\OneDrive\Desktop\Innomatics\MLOps Project\reviews_data_dump\reviews_badminton\data.csv"
df = pd.read_csv(DATA_PATH)

df = df[df["Ratings"] != 3]
df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)
df["text"] = df["Review Title"].fillna("") + " " + df["Review text"].fillna("")
df["processed_text"] = df["text"].apply(preprocess)

X = df["processed_text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# ========================
# MODELS
# ========================
models = {
    "LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000)),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "LinearSVC": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000)),
        ("model", LinearSVC())
    ]),
    "NaiveBayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000)),
        ("model", MultinomialNB())
    ]),
    "RandomForest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        ))
    ])
}

results = {}

# ========================
# TRAIN ML MODELS
# ========================
for name, model in models.items():
    print(f"\n--- Training {name} ---")

    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test

    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "f1_score": f1,
        "fit_time": fit_time,
        "test_time": test_time
    }

    print(
        f"{name} | TrainAcc={train_acc:.4f} | "
        f"TestAcc={test_acc:.4f} | F1={f1:.4f}"
    )

# ========================
# LSTM MODEL
# ========================
print("\n--- Training LSTM ---")

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=150)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=150)

lstm_model = Sequential([
    Embedding(10000, 128, input_length=150),
    LSTM(128),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

start_fit = time.time()
lstm_model.fit(X_train_pad, y_train, epochs=3, batch_size=64, verbose=0)
fit_time = time.time() - start_fit

start_test = time.time()
y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype(int).ravel()
test_time = time.time() - start_test

train_acc = lstm_model.evaluate(X_train_pad, y_train, verbose=0)[1]
test_acc = accuracy_score(y_test, y_pred_lstm)
f1 = f1_score(y_test, y_pred_lstm)

results["LSTM"] = {
    "model": lstm_model,
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "f1_score": f1,
    "fit_time": fit_time,
    "test_time": test_time
}

print(
    f"LSTM | TrainAcc={train_acc:.4f} | "
    f"TestAcc={test_acc:.4f} | F1={f1:.4f}"
)

# ========================
# SELECT BEST MODEL (F1)
# ========================
best_model_name = max(results, key=lambda x: results[x]["f1_score"])
best_model_info = results[best_model_name]

print("\n================ BEST MODEL ================")
print(f"🏆 Model: {best_model_name}")
print(f"🎯 F1 Score: {best_model_info['f1_score']:.4f}")
print("===========================================\n")

# ========================
# SAVE BEST MODEL
# ========================
if best_model_name == "LSTM":
    best_model_info["model"].save("best_sentiment_model_lstm.h5")
    joblib.dump(tokenizer, "best_lstm_tokenizer.pkl")
else:
    joblib.dump(best_model_info["model"], "best_sentiment_model.pkl")

print("✅ Best model saved successfully.")
