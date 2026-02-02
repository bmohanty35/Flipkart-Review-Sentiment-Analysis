# =====================================================
# Sentiment Analysis | ML + LSTM
# Best Model Selection using F1 Score (FIXED)
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
# NLP SETUP (FIXED)
# ========================
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 🔥 REMOVE NEGATION WORDS FROM STOPWORDS
STOP_WORDS = set(stopwords.words("english")) - {"not", "no", "nor"}
LEMMATIZER = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)

    # NEGATION HANDLING (CRITICAL)
    text = re.sub(r"\bnot\s+(\w+)", r"not_\1", text)

    # KEEP UNDERSCORE
    text = re.sub(r"[^a-z_\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = [LEMMATIZER.lemmatize(w) for w in text.split() if w not in STOP_WORDS]
    return " ".join(words)

print(preprocess("the product is not worth"))


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
# MODELS (FIXED TF-IDF)
# ========================
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 3),     # 🔥 TRIGRAMS
    min_df=5
)

models = {
    "LogisticRegression": Pipeline([
        ("tfidf", tfidf),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "LinearSVC": Pipeline([
        ("tfidf", tfidf),
        ("model", LinearSVC())
    ]),
    "NaiveBayes": Pipeline([
        ("tfidf", tfidf),
        ("model", MultinomialNB())
    ]),
    "RandomForest": Pipeline([
        ("tfidf", tfidf),
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

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": model,
        "f1_score": f1
    }

    print(f"{name} | Train={train_acc:.4f} | Test={test_acc:.4f} | F1={f1:.4f}")

# ========================
# LSTM MODEL (UNCHANGED)
# ========================
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

lstm_model.fit(X_train_pad, y_train, epochs=3, batch_size=64, verbose=0)

y_pred_lstm = (lstm_model.predict(X_test_pad) > 0.5).astype(int).ravel()
f1_lstm = f1_score(y_test, y_pred_lstm)

results["LSTM"] = {
    "model": lstm_model,
    "f1_score": f1_lstm
}

print(f"LSTM | F1={f1_lstm:.4f}")

# ========================
# SELECT & SAVE BEST MODEL
# ========================
# ========================
# FORCE LinearSVC AS BEST MODEL
# ========================

best_model_name = "LinearSVC"
best_model = results["LinearSVC"]["model"]

print("\n🏆 BEST MODEL (FORCED): LinearSVC")
print(f"🎯 F1 Score: {results['LinearSVC']['f1_score']:.4f}")

# ========================
# SAVE MODEL
# ========================
joblib.dump(best_model, "best_sentiment_model.pkl")

print("✅ LinearSVC model saved successfully as best_sentiment_model.pkl")
