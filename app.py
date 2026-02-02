import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ================================
# NLP SETUP (MUST MATCH TRAINING)
# ================================
nltk.download("stopwords")
nltk.download("wordnet")

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

# ================================
# Load trained model
# ================================
model = joblib.load("best_sentiment_model.pkl")

# ================================
# Streamlit UI
# ================================
st.title("Flipkart Review Sentiment Analysis")
st.write("Enter a product review to predict sentiment")

review = st.text_area(
    "Customer Review",
    placeholder="Type or paste a Flipkart product review here..."
)

if st.button("Predict Sentiment"):
    if review.strip():
        processed_review = preprocess(review)
        prediction = model.predict([processed_review])[0]

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review")
