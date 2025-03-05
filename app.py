from flask import Flask, render_template, request, send_file
import lightgbm as lgb
from lightgbm import LGBMClassifier
import joblib
import numpy as np
import pandas as pd
import re
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

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
    "very", "extremely", "so", "quite", "really", "almost", "just"}

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

# Home page route
@app.route('/', methods=['GET', 'POST'])
def home():
    processed_text = ""
    model_choice = "lightgbm"
    result = None
    Comment = ""
    if request.method == 'POST':
        if 'Comment' in request.form:
            Comment = request.form['Comment']
            model_choice = request.form.get('model', 'lightgbm')

            Processed_Comments = preprocess_text(Comment)  
            if Processed_Comments != "empty_Comment":
                tfidf_vector = tfidf_vectorizer.transform([Processed_Comments])

                if model_choice == 'random_forest':
                    prediction = rf_model.predict(tfidf_vector)
                else:
                    prediction = lgbm_model.predict(tfidf_vector)
                    prediction = np.argmax(prediction, axis=1)[0]

                result = get_Sentiment_Label(prediction)

        elif 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.filename.endswith('.json'):
                df = pd.read_json(file)
            else:
                return "Invalid file format", 400

            df['Processed_Comments'] = df['Comment'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else "empty_Comment")
            df = df[df['Processed_Comments'] != "empty_Comment"]

            df['Sentiment_Label'] = df['Processed_Comments'].apply(
                lambda x: get_Sentiment_Label(lgbm_model.predict(tfidf_vectorizer.transform([x]))))

            result_file = 'static/results/results.csv'
            df.to_csv(result_file, index=False)
            result = df['Sentiment_Label'].value_counts().idxmax()

            return send_file(result_file, as_attachment=True)

    return render_template('index.html', 
            result=result, 
            processed_text=processed_text,
            model_choice=model_choice, 
            original_comment = Comment)

if __name__ == '__main__':
    app.run(debug=True)


