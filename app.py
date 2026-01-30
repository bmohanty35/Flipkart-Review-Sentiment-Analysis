import streamlit as st
import joblib

# ================================
# Load trained model
# ================================
model = joblib.load("sentiment_model.pkl")

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
        prediction = model.predict([review])[0]

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")
    else:
        st.warning("Please enter a review")
