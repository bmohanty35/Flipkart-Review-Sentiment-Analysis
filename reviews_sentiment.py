import pandas as pd
import re
import nltk
import joblib
import warnings
warnings.filterwarnings("ignore")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

# ================================
# NLTK setup
# ================================
nltk.download("stopwords")
nltk.download("wordnet")

# ================================
# Load Dataset
# ================================
df = pd.read_csv(
    r"C:\Users\bmoha\OneDrive\Desktop\Innomatics\MLOps Project\reviews_data_dump\reviews_badminton\data.csv"
)

# 2. Label Creation
# ================================
df = df[df["Ratings"] != 3]
df["sentiment"] = df["Ratings"].apply(lambda x: 1 if x >= 4 else 0)

df["text"] = df["Review Title"].fillna("") + " " + df["Review text"].fillna("")

# ================================
# Text Preprocessing
# ================================
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = " ".join(w for w in text.split() if w not in stop_words)
    text = " ".join(lemmatizer.lemmatize(w) for w in text.split())
    return text

df["processed_text"] = df["text"].apply(preprocess)

# ================================
# Train–Test Split
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    df["processed_text"],
    df["sentiment"],
    test_size=0.25,
    random_state=42,
    stratify=df["sentiment"]
)

# ================================
# Pipeline (ONLY sklearn objects)
# ================================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=7000, ngram_range=(1, 2))),
    ("model", LinearSVC())
])

# ================================
# Train Model
# ================================
pipeline.fit(X_train, y_train)

# ================================
# Evaluate
# ================================
y_pred = pipeline.predict(X_test)
print("F1-score:", f1_score(y_test, y_pred))

# ================================
# Save Model
# ================================
joblib.dump(pipeline, "sentiment_model.pkl")
print("✅ sentiment_model.pkl created successfully")
