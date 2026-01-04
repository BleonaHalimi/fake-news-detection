#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train and Save Models Script
This script trains all 4 ML models and the TF-IDF vectorizer,
then saves them as pickle files for fast loading in the Streamlit app.

Run this script once before starting the Streamlit application.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string
import joblib
import json
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from config import APP_CONFIG, PATHS, MODEL_FILES

print("="*70)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("="*70)

# Text preprocessing function
def wordopt(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Step 1: Load datasets
print("\n[1/8] Loading datasets...")
df_fake = pd.read_csv(f"{PATHS['data_dir']}Fake.csv")
df_true = pd.read_csv(f"{PATHS['data_dir']}True.csv")
print(f"  [OK] Loaded {len(df_fake):,} fake news articles")
print(f"  [OK] Loaded {len(df_true):,} true news articles")

# Step 2: Add class labels
print("\n[2/8] Adding class labels...")
df_fake["class"] = 0
df_true["class"] = 1
print("  [OK] Labels added (0=Fake, 1=True)")

# Step 3: Remove last 10 rows for manual testing
print("\n[3/8] Extracting manual testing data...")
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]
print(f"  [OK] Training set: {len(df_fake):,} fake, {len(df_true):,} true")

# Step 4: Merge and prepare data
print("\n[4/8] Merging and preparing data...")
df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1, random_state=APP_CONFIG['random_state'])
df.reset_index(drop=True, inplace=True)
print(f"  [OK] Total articles: {len(df):,}")

# Step 5: Save original text and preprocess
print("\n[5/8] Preprocessing text data...")
print("  This may take a few minutes...")
df["text_original"] = df["text"]  # Save original text for feature extraction
df["text"] = df["text"].apply(wordopt)  # Preprocessed text for TF-IDF
print("  [OK] Text preprocessing complete")

# Step 6: Split data and vectorize
print("\n[6/8] Splitting data and vectorizing...")
x = df["text"]
x_orig = df["text_original"]  # Original text for advanced features
y = df["class"]
x_train, x_test, x_train_orig, x_test_orig, y_train, y_test = train_test_split(
    x, x_orig, y,
    test_size=APP_CONFIG['test_size'],
    random_state=APP_CONFIG['random_state']
)
print(f"  [OK] Training samples: {len(x_train):,}")
print(f"  [OK] Testing samples: {len(x_test):,}")

# Vectorize with bigrams
print("\n  Creating TF-IDF vectorizer with bigrams...")
vectorizer = TfidfVectorizer(
    max_features=APP_CONFIG['tfidf_max_features'],
    ngram_range=(1, 2),  # Include unigrams and bigrams
    min_df=2,            # Ignore terms appearing in < 2 documents
    max_df=0.95          # Ignore terms appearing in > 95% of documents
)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)
print(f"  [OK] TF-IDF features: {xv_train.shape[1]}")

# Extract advanced features
print("\n  Extracting advanced linguistic features...")
from utils.text_preprocessing import extract_advanced_features, features_to_array
from scipy.sparse import hstack
import numpy as np

# Extract features for all texts
print("    Processing training data...")
train_features = []
for idx, text in enumerate(x_train_orig):
    feats = extract_advanced_features(text)
    train_features.append(features_to_array(feats))
    if (idx + 1) % 5000 == 0:
        print(f"      Processed {idx + 1}/{len(x_train_orig)} articles...")
train_features = np.array(train_features)

print("    Processing test data...")
test_features = []
for idx, text in enumerate(x_test_orig):
    feats = extract_advanced_features(text)
    test_features.append(features_to_array(feats))
    if (idx + 1) % 2000 == 0:
        print(f"      Processed {idx + 1}/{len(x_test_orig)} articles...")
test_features = np.array(test_features)

print(f"  [OK] Advanced features: {train_features.shape[1]}")

# Combine TF-IDF with advanced features
print("\n  Combining TF-IDF and advanced features...")
xv_train = hstack([xv_train, train_features])
xv_test = hstack([xv_test, test_features])
print(f"  [OK] Total feature dimension: {xv_train.shape[1]}")

# Step 7: Train all models
print("\n[7/8] Training machine learning models...")
models = {}
accuracies = {}

# Logistic Regression
print("\n  [1/4] Training Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=APP_CONFIG['random_state'])
lr.fit(xv_train, y_train)
lr_score = lr.score(xv_test, y_test)
models['lr'] = lr
accuracies['Logistic Regression'] = round(lr_score * 100, 2)
print(f"    [OK] Accuracy: {lr_score*100:.2f}%")

# Decision Tree
print("\n  [2/4] Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=APP_CONFIG['random_state'])
dt.fit(xv_train, y_train)
dt_score = dt.score(xv_test, y_test)
models['dt'] = dt
accuracies['Decision Tree'] = round(dt_score * 100, 2)
print(f"    [OK] Accuracy: {dt_score*100:.2f}%")

# Gradient Boosting
print("\n  [3/4] Training Gradient Boosting Classifier...")
gbc = GradientBoostingClassifier(
    random_state=APP_CONFIG['random_state'],
    n_estimators=100
)
gbc.fit(xv_train, y_train)
gbc_score = gbc.score(xv_test, y_test)
models['gbc'] = gbc
accuracies['Gradient Boosting'] = round(gbc_score * 100, 2)
print(f"    [OK] Accuracy: {gbc_score*100:.2f}%")

# Random Forest
print("\n  [4/4] Training Random Forest...")
rfc = RandomForestClassifier(
    random_state=APP_CONFIG['random_state'],
    n_estimators=100
)
rfc.fit(xv_train, y_train)
rfc_score = rfc.score(xv_test, y_test)
models['rfc'] = rfc
accuracies['Random Forest'] = round(rfc_score * 100, 2)
print(f"    [OK] Accuracy: {rfc_score*100:.2f}%")

# Step 8: Save models and metadata
print("\n[8/8] Saving models and metadata...")

# Create models directory if it doesn't exist
os.makedirs(PATHS['models_dir'], exist_ok=True)

# Save each model
print(f"\n  Saving to {PATHS['models_dir']}")
for model_key, model in models.items():
    filepath = f"{PATHS['models_dir']}{MODEL_FILES[model_key]}"
    joblib.dump(model, filepath)
    print(f"  [OK] Saved {MODEL_FILES[model_key]}")

# Save vectorizer
vectorizer_path = f"{PATHS['models_dir']}{MODEL_FILES['vectorizer']}"
joblib.dump(vectorizer, vectorizer_path)
print(f"  [OK] Saved {MODEL_FILES['vectorizer']}")

# Save metadata
metadata = {
    'training_date': datetime.now().isoformat(),
    'accuracies': accuracies,
    'training_samples': len(x_train),
    'testing_samples': len(x_test),
    'tfidf_features': APP_CONFIG['tfidf_max_features'],
    'random_state': APP_CONFIG['random_state']
}

metadata_path = f"{PATHS['models_dir']}{MODEL_FILES['metadata']}"
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  [OK] Saved {MODEL_FILES['metadata']}")

# Display summary
print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print("\nModel Performance Summary:")
print("-"*70)
for model_name, accuracy in accuracies.items():
    print(f"  {model_name:<25} {accuracy:>6.2f}%")
print("-"*70)
print(f"\nAll models saved to: {PATHS['models_dir']}")
print("\nYou can now run the Streamlit app with:")
print("  streamlit run app.py")
print("="*70)
