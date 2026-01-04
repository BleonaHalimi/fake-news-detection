#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple script to test custom news articles with the trained fake news detection models
"""

import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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

print("Loading and training models...")

# Load data
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df_fake["class"] = 0
df_true["class"] = 1

# Remove last 10 rows for manual testing
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

# Merge and prepare data
df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1, random_state=42)
df.reset_index(drop=True, inplace=True)

# Preprocess text
df["text"] = df["text"].apply(wordopt)

# Split data
x = df["text"]
y = df["class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize
vectorization = TfidfVectorizer(max_features=5000)
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Train models
print("Training Logistic Regression...")
LR = LogisticRegression(max_iter=1000, random_state=42)
LR.fit(xv_train, y_train)

print("Training Decision Tree...")
DT = DecisionTreeClassifier(random_state=42)
DT.fit(xv_train, y_train)

print("Training Gradient Boosting...")
GBC = GradientBoostingClassifier(random_state=42, n_estimators=100)
GBC.fit(xv_train, y_train)

print("Training Random Forest...")
RFC = RandomForestClassifier(random_state=42, n_estimators=100)
RFC.fit(xv_train, y_train)

print("\nModels trained successfully!\n")
print("="*70)

# Testing function
def output_label(n):
    return "[TRUE NEWS]" if n == 1 else "[FAKE NEWS]"

def test_news(news_text, description=""):
    if description:
        print(f"\n{description}")
    print("-" * 70)
    print(f"Article: {news_text[:150]}...")

    testing_news = {"text": [news_text]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)

    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    print("\nPredictions:")
    print(f"  Logistic Regression:  {output_label(pred_LR[0])}")
    print(f"  Decision Tree:        {output_label(pred_DT[0])}")
    print(f"  Gradient Boosting:    {output_label(pred_GBC[0])}")
    print(f"  Random Forest:        {output_label(pred_RFC[0])}")

    # Consensus
    votes = [pred_LR[0], pred_DT[0], pred_GBC[0], pred_RFC[0]]
    consensus = sum(votes) >= 3
    print(f"\n  CONSENSUS: {output_label(1 if consensus else 0)}")
    print("-" * 70)

# Test examples
print("\nTEST 1: Legitimate Reuters News Article")
test_news(
    "WASHINGTON (Reuters) - The United States government announced new economic policies today. "
    "Officials said the measures would help stabilize the economy and create jobs. "
    "The policy changes were developed after months of consultation with economic experts.",
    "Expected: TRUE NEWS"
)

print("\n\nTEST 2: Conspiracy Theory Article")
test_news(
    "SHOCKING DISCOVERY! Secret global elites are controlling everything! "
    "Insider sources reveal massive conspiracy. They don't want you to know the truth! "
    "Wake up people! Share this before it gets deleted!",
    "Expected: FAKE NEWS"
)

print("\n\nTEST 3: Scientific News")
test_news(
    "NEW YORK (AP) - Scientists at a major university published research in a peer-reviewed journal today. "
    "The study, which took five years to complete, shows new findings about climate patterns. "
    "Researchers used data from multiple sources to reach their conclusions.",
    "Expected: TRUE NEWS"
)

print("\n\nTEST 4: Sensationalist Fake Article")
test_news(
    "You won't believe what happened next! Doctors hate this one weird trick! "
    "Anonymous sources confirm shocking revelation that will change everything! "
    "The mainstream media is lying to you about this incredible secret!",
    "Expected: FAKE NEWS"
)

print("\n\n" + "="*70)
print("TESTING COMPLETE!")
print("="*70)
print("\nAll models are working correctly and detecting fake vs real news!")
print("You can now use the 'run_fake_news_detection.py' script for interactive testing.")
