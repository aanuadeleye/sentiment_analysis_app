import streamlit as st
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize NLP tools
lemmatizer = WordNetLemmatizer()

# Load trained models
lgbm_model = lgb.Booster(model_file='models/lgbm_model.txt')
rf_model = joblib.load('models/random_forest_model.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Function to get sentiment label
def get_Sentiment_Label(prediction):
    pred_class = np.argmax(prediction) if isinstance(prediction, np.ndarray) else int(prediction)
    return ["Negative", "Neutral", "Positive"][pred_class]  # 0 -> Negative, 1 -> Neutral, 2 -> Positive

# Preprocessing function
def preprocess_text(text):
    valuable_sentiment_words = {
        "not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere",
        "can't", "cannot", "won't", "don't", "didn't", "shouldn't", "wouldn't", "couldn't",
        "isn't", "aren't", "wasn't", "weren't",
        "but", "however",
        "very", "extremely", "so", "quite", "really", "almost", "just"
    }

    default_stop_words = set(stopwords.words('english'))
    custom_stop_words = default_stop_words.difference(valuable_sentiment_words)
    
    # Ensure that text is a string and handle NaN (float) values
    if isinstance(text, str):
        text = text.lower()  # Perform string-specific processing
    else:
        text = str(text)  # Convert float or NaN to string before processing
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stop_words and word.isalnum()]
    return ' '.join(tokens) if tokens else "empty_Comment"

# Streamlit App
st.title("Sentiment Analysis App")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose Model", ["LightGBM", "Random Forest"])

# Text input for single comment analysis
st.subheader("Analyze Single Comment")
Comment = st.text_area("Enter your comment here:", height=150)

if st.button("Analyze Sentiment"):
    if Comment.strip() == "":
        st.warning("Please enter a comment to analyze.")
    else:
        Processed_Comments = preprocess_text(Comment)
        if Processed_Comments != "empty_Comment":
            tfidf_vector = tfidf_vectorizer.transform([Processed_Comments])

            if model_choice == "Random Forest":
                prediction = rf_model.predict(tfidf_vector)
            else:
                prediction = lgbm_model.predict(tfidf_vector)
                prediction = np.argmax(prediction, axis=1)[0]

            result = get_Sentiment_Label(prediction)
            st.subheader("Sentiment Analysis Result:")
            st.write(f"**Sentiment**: {result}")
        else:
            st.error("The comment could not be processed.")

# File upload for batch analysis
st.subheader("Batch Analysis")
uploaded_file = st.file_uploader("Upload a CSV or JSON file", type=["csv", "json"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        
        df['Processed_Comments'] = df['Comment'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else "empty_Comment")
        df = df[df['Processed_Comments'] != "empty_Comment"]

        df['Sentiment_Label'] = df['Processed_Comments'].apply(
            lambda x: get_Sentiment_Label(lgbm_model.predict(tfidf_vectorizer.transform([x]))))

        st.subheader("Batch Analysis Results")
        st.write(df)

        # Download results as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
