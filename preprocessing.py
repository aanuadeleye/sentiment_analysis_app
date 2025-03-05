import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
import pickle 

# Load dataset
df = pd.read_csv("/Users/seun/project/sentiment_analysis/datasets/netflix_fb_comments.csv")

# Function to get POS tag for better lemmatization
def get_wordnet_pos(word):
    """
    Map POS tag to first character for WordNetLemmatizer.
    """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Function to clean text
def preprocess_text(text):
    """
    Preprocess text by performing:
    1. Lowercasing
    2. URL removal
    3. Tokenization
    4. Stopword removal
    5. Lemmatization
    6. Punctuation and special character removal
    """
    if pd.isna(text):  # Handle NaN values
        return ""

    # Define a set of sentiment-bearing words that you want to preserve
    valuable_sentiment_words = {
    # Negations
    "not", "no", "nor", "never", "none", "nobody", "nothing", "neither", "nowhere",
    "can't", "cannot", "won't", "don't", "didn't", "shouldn't", "wouldn't", "couldn't",
    "isn't", "aren't", "wasn't", "weren't",

    # Contrast words
    "but", "however",

    # Intensifiers / Modifiers
    "very", "extremely", "so", "quite", "really", "almost", "just"}

    # Load the default stop words from NLTK
    default_stop_words = set(stopwords.words('english'))

    # Create a custom stop words set by subtracting the valuable sentiment words
    custom_stop_words = default_stop_words.difference(valuable_sentiment_words)

    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove URLs, mentions (@), and hashtags (#)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)

    # 3. Remove numbers
    text = re.sub(r"\d+", "", text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in custom_stop_words]

    # 6. POS-based lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]

    # 7. Join tokens back into a single string
    return " ".join(tokens)

# Apply preprocessing to the "Comment" column
df["Processed_Comments"] = df["Comment"].apply(preprocess_text)

# **STRATIFIED SAMPLING IMPLEMENTATION**
# Define the number of samples per class (adjust based on dataset size and balance requirements)
num_samples_per_class = 25000  # Adjust as needed

# Apply stratified sampling to balance the dataset
stratified_df = df.groupby("Sentiment_Label", group_keys=False).apply(lambda x: x.sample(min(len(x), num_samples_per_class)))

# Split dataset into training (70%), testing (15%), and validation (15%)
train_df, temp_df = train_test_split(
    stratified_df, test_size=0.3, stratify=stratified_df['Sentiment_Label'], random_state=42
)
test_df, val_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df['Sentiment_Label'], random_state=42
)

# Save the processed and stratified dataset
train_df.to_csv("/Users/seun/project/sentiment_analysis/datasets/train_dataset.csv", index=False)
test_df.to_csv("/Users/seun/project/sentiment_analysis/datasets/test_dataset.csv", index=False)
val_df.to_csv("/Users/seun/project/sentiment_analysis/datasets/validation_dataset.csv", index=False)

print("Preprocessing complete! Train, test, and validation data saved.")
