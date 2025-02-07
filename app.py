import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_vectorizer_and_model():
    vectorizer = joblib.load("tfidf_vectorizer.joblib")  
    classifier = joblib.load("random_forest_classifier.joblib")  
    return vectorizer, classifier

vectorizer, classifier = load_vectorizer_and_model()

st.title("Spam Detection System")
st.write("This app uses a machine learning model to classify text messages as **Spam** or **Not Spam**.")

user_input = st.text_area("Enter a message:", placeholder="Type your message here...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        
        user_input_vectorized = vectorizer.transform([user_input])
    
        prediction = classifier.predict(user_input_vectorized)[0]
        prediction_proba = classifier.predict_proba(user_input_vectorized)[0]
        
        if prediction == 1:
            st.error("The message is classified as **Spam** hutt.")
        else:
            st.success("The message is classified as **Not Spam**.")

        st.write(f"**Confidence:**")
        st.write(f"- Spam: {prediction_proba[1] * 100:.2f}%")
        st.write(f"- Not Spam: {prediction_proba[0] * 100:.2f}%")

st.sidebar.header("About")
st.sidebar.write("""
- This app uses a machine learning model trained with **TfidfVectorizer** and **MultinomialNB**.
- Built with **Streamlit** for spam detection tasks.
- You can enter any message, and the app will classify it as Spam or Not Spam with a confidence score.

Commit by zahid
ff
f
f

""")

