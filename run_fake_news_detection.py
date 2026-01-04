#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fake News Detection Script
This script trains multiple ML models to detect fake news and allows testing with custom articles.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FAKE NEWS DETECTION SYSTEM")
print("="*60)

# Load datasets
print("\n[1/10] Loading datasets...")
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
print(f"  - Loaded {len(df_fake)} fake news articles")
print(f"  - Loaded {len(df_true)} true news articles")

# Add class labels
print("\n[2/10] Adding class labels...")
df_fake["class"] = 0
df_true["class"] = 1

# Extract manual testing data
print("\n[3/10] Extracting manual testing data...")
df_fake_manual_testing = df_fake.tail(10).copy()
indices_to_drop = df_fake.tail(10).index.tolist()
df_fake.drop(indices_to_drop, axis=0, inplace=True)

df_true_manual_testing = df_true.tail(10).copy()
indices_to_drop = df_true.tail(10).index.tolist()
df_true.drop(indices_to_drop, axis=0, inplace=True)

df_manual_testing = pd.concat([df_fake_manual_testing, df_true_manual_testing], axis=0)
df_manual_testing.to_csv("manual_testing.csv", index=False)
print(f"  - Saved {len(df_manual_testing)} articles for manual testing")

# Merge datasets
print("\n[4/10] Merging datasets...")
df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)
print(f"  - Total articles: {len(df)}")

# Shuffle and reset index
print("\n[5/10] Shuffling data...")
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply text preprocessing
print("\n[6/10] Preprocessing text data...")
df["text"] = df["text"].apply(wordopt)

# Split data
print("\n[7/10] Splitting data into training and testing sets...")
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
print(f"  - Training samples: {len(x_train)}")
print(f"  - Testing samples: {len(x_test)}")

# Vectorization
print("\n[8/10] Vectorizing text using TF-IDF...")
vectorization = TfidfVectorizer(max_features=5000)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
print(f"  - Feature dimension: {xv_train.shape[1]}")

# Train models
print("\n[9/10] Training machine learning models...")

# Logistic Regression
print("  - Training Logistic Regression...")
LR = LogisticRegression(max_iter=1000, random_state=42)
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
lr_score = LR.score(xv_test, y_test)
print(f"    Accuracy: {lr_score:.4f}")

# Decision Tree
print("  - Training Decision Tree...")
DT = DecisionTreeClassifier(random_state=42)
DT.fit(xv_train, y_train)
pred_dt = DT.predict(xv_test)
dt_score = DT.score(xv_test, y_test)
print(f"    Accuracy: {dt_score:.4f}")

# Gradient Boosting
print("  - Training Gradient Boosting Classifier...")
GBC = GradientBoostingClassifier(random_state=42, n_estimators=100)
GBC.fit(xv_train, y_train)
pred_gbc = GBC.predict(xv_test)
gbc_score = GBC.score(xv_test, y_test)
print(f"    Accuracy: {gbc_score:.4f}")

# Random Forest
print("  - Training Random Forest...")
RFC = RandomForestClassifier(random_state=42, n_estimators=100)
RFC.fit(xv_train, y_train)
pred_rfc = RFC.predict(xv_test)
rfc_score = RFC.score(xv_test, y_test)
print(f"    Accuracy: {rfc_score:.4f}")

# Model comparison
print("\n[10/10] Model Performance Summary:")
print("="*60)
print(f"{'Model':<30} {'Accuracy':<15}")
print("-"*60)
print(f"{'Logistic Regression':<30} {lr_score:.4f}")
print(f"{'Decision Tree':<30} {dt_score:.4f}")
print(f"{'Gradient Boosting':<30} {gbc_score:.4f}")
print(f"{'Random Forest':<30} {rfc_score:.4f}")
print("="*60)

# Testing function
def output_label(n):
    if n == 0:
        return "FAKE NEWS"
    elif n == 1:
        return "TRUE NEWS"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Logistic Regression:         {output_label(pred_LR[0])}")
    print(f"Decision Tree:               {output_label(pred_DT[0])}")
    print(f"Gradient Boosting:           {output_label(pred_GBC[0])}")
    print(f"Random Forest:               {output_label(pred_RFC[0])}")
    print("="*60)

# Interactive testing
print("\n" + "="*60)
print("FAKE NEWS DETECTION - INTERACTIVE TESTING")
print("="*60)
print("\nYou can now test the models with custom news articles.")
print("Type 'quit' or 'exit' to stop.\n")

# Test with some examples
test_examples = [
    {
        "label": "Example 1 (Should be TRUE NEWS)",
        "text": "BRUSSELS (Reuters) - NATO allies on Tuesday welcomed President Donald Trump s decision to commit more forces to Afghanistan, as part of a new U.S. strategy he said would require more troops and funding from America s partners."
    },
    {
        "label": "Example 2 (Should be FAKE NEWS)",
        "text": "Vic Bishop Waking TimesOur reality is carefully constructed by powerful corporate, political and special interest sources in order to covertly sway public opinion. Blatant lies are often televised regarding terrorism, food, war, health, etc."
    }
]

for example in test_examples:
    print(f"\nTesting: {example['label']}")
    print(f"Article preview: {example['text'][:100]}...")
    manual_testing(example['text'])

# Interactive mode
while True:
    print("\n" + "-"*60)
    user_input = input("\nEnter a news article to test (or 'quit' to exit): ").strip()

    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nThank you for using the Fake News Detection System!")
        break

    if len(user_input) < 50:
        print("Please enter a longer article (at least 50 characters) for better prediction accuracy.")
        continue

    manual_testing(user_input)

print("\n" + "="*60)
print("Training complete! Models are ready for fake news detection.")
print("="*60)
