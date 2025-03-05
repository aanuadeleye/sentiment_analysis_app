import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load preprocessed datasets
train_data = pd.read_csv('/Users/seun/project/main/datasets/train_dataset.csv')
val_data = pd.read_csv('/Users/seun/project/main/datasets/validation_dataset.csv')
test_data = pd.read_csv('/Users/seun/project/main/datasets/test_dataset.csv')

# Extract features and labels
X_train_text = train_data['Processed_Comments']
y_train = train_data['Sentiment_Label'].astype(int)
X_val_text = val_data['Processed_Comments']
y_val = val_data['Sentiment_Label'].astype(int)
X_test_text = test_data['Processed_Comments']
y_test = test_data['Sentiment_Label'].astype(int)

# Map sentiment labels: -1 -> 0 (negative), 0 -> 1 (neutral), 1 -> 2 (positive)
label_mapping = {-1: 0, 0: 1, 1: 2}
y_train = y_train.map(label_mapping)
y_val = y_val.map(label_mapping)
y_test = y_test.map(label_mapping)

# Apply TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train = tfidf_vectorizer.fit_transform(X_train_text).toarray()
X_val = tfidf_vectorizer.transform(X_val_text).toarray()
X_test = tfidf_vectorizer.transform(X_test_text).toarray()

# Save TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

# Train LightGBM model
train_dataset = lgb.Dataset(X_train, label=y_train)
val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

params = {
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 3,
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

lgbm_model = lgb.train(params, train_dataset, num_boost_round=100, valid_sets=[val_dataset])

# Predict with LightGBM
y_pred_prob_lgbm = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
y_pred_lgbm = np.argmax(y_pred_prob_lgbm, axis=1)

# Train Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict with Random Forest
y_pred_rf = rf_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} Accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"{model_name} Precision: {precision_score(y_true, y_pred, average='weighted')}")
    print(f"{model_name} Recall: {recall_score(y_true, y_pred, average='weighted')}")
    print(f"{model_name} F1-Score: {f1_score(y_true, y_pred, average='weighted')}")
    print(classification_report(y_true, y_pred, target_names=['negative', 'neutral', 'positive']))

evaluate_model(y_test, y_pred_lgbm, "LightGBM")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Save models
lgbm_model.save_model("lgbm_model.txt")
joblib.dump(rf_model, "random_forest_model.pkl")
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")
