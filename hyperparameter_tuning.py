#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hyperparameter Tuning Script
Optimizes model hyperparameters using GridSearchCV
This script finds the best parameters for Random Forest and Gradient Boosting models.

Run this script to improve model performance beyond default parameters.
Note: This can take 30-60 minutes depending on hardware.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import re
import string
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import APP_CONFIG, PATHS, MODEL_FILES
from utils.text_preprocessing import wordopt, extract_advanced_features, features_to_array
from scipy.sparse import hstack

print("="*70)
print("FAKE NEWS DETECTION - HYPERPARAMETER TUNING")
print("="*70)
print(f"\nTuning started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nWARNING: This process may take 30-60 minutes!")
print("Using GridSearchCV with 3-fold cross-validation")

# Step 1: Load and prepare data
print("\n[1/5] Loading and preparing data...")
df_fake = pd.read_csv(f"{PATHS['data_dir']}Fake.csv")
df_true = pd.read_csv(f"{PATHS['data_dir']}True.csv")

df_fake["class"] = 0
df_true["class"] = 1
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

df_merge = pd.concat([df_fake, df_true], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)
df = df.sample(frac=1, random_state=APP_CONFIG['random_state'])
df.reset_index(drop=True, inplace=True)
print(f"  [OK] Total articles: {len(df):,}")

# Step 2: Preprocess
print("\n[2/5] Preprocessing text...")
df["text_original"] = df["text"]
df["text"] = df["text"].apply(wordopt)
print("  [OK] Text preprocessing complete")

# Step 3: Create features
print("\n[3/5] Creating feature vectors...")
x = df["text"]
x_orig = df["text_original"]
y = df["class"]

# Split data
x_train, x_test, x_train_orig, x_test_orig, y_train, y_test = train_test_split(
    x, x_orig, y,
    test_size=APP_CONFIG['test_size'],
    random_state=APP_CONFIG['random_state']
)

# Vectorize with bigrams
print("  Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=APP_CONFIG['tfidf_max_features'],
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Extract advanced features
print("  Extracting advanced features...")
print("    Processing training data...")
train_features = []
for idx, text in enumerate(x_train_orig):
    feats = extract_advanced_features(text)
    train_features.append(features_to_array(feats))
    if (idx + 1) % 5000 == 0:
        print(f"      {idx + 1}/{len(x_train_orig)} processed...")
train_features = np.array(train_features)

print("    Processing test data...")
test_features = []
for idx, text in enumerate(x_test_orig):
    feats = extract_advanced_features(text)
    test_features.append(features_to_array(feats))
test_features = np.array(test_features)

# Combine features
xv_train = hstack([xv_train, train_features])
xv_test = hstack([xv_test, test_features])
print(f"  [OK] Total feature dimension: {xv_train.shape[1]}")

# Step 4: Hyperparameter Tuning
print("\n[4/5] Running hyperparameter tuning...")

results = {}

# Random Forest Tuning
print("\n  [1/2] Tuning Random Forest Classifier...")
print("    This will test different combinations of parameters...")

rf_param_grid = {
    'n_estimators': [100, 200],           # Number of trees
    'max_depth': [None, 50, 100],         # Maximum tree depth
    'min_samples_split': [2, 5],          # Minimum samples to split node
    'min_samples_leaf': [1, 2],           # Minimum samples in leaf
    'max_features': ['sqrt', 'log2']      # Number of features per split
}

rf_base = RandomForestClassifier(random_state=APP_CONFIG['random_state'], n_jobs=-1)

rf_grid = GridSearchCV(
    rf_base,
    rf_param_grid,
    cv=3,                    # 3-fold CV for speed
    scoring='accuracy',
    n_jobs=-1,              # Use all CPU cores
    verbose=1,
    return_train_score=True
)

print("    Starting grid search (this may take 20-30 minutes)...")
rf_grid.fit(xv_train, y_train)

# Get best model
best_rf = rf_grid.best_estimator_
rf_score = best_rf.score(xv_test, y_test)

results['Random Forest'] = {
    'best_params': rf_grid.best_params_,
    'best_cv_score': rf_grid.best_score_,
    'test_accuracy': rf_score,
    'default_accuracy': RandomForestClassifier(
        random_state=APP_CONFIG['random_state'],
        n_estimators=100
    ).fit(xv_train, y_train).score(xv_test, y_test)
}

print(f"\n    [RESULTS]")
print(f"      Best Parameters: {rf_grid.best_params_}")
print(f"      Best CV Score: {rf_grid.best_score_*100:.2f}%")
print(f"      Test Accuracy: {rf_score*100:.2f}%")
print(f"      Improvement over default: {(rf_score - results['Random Forest']['default_accuracy'])*100:+.2f}%")

# Gradient Boosting Tuning
print("\n  [2/2] Tuning Gradient Boosting Classifier...")
print("    This will test different combinations of parameters...")

gb_param_grid = {
    'n_estimators': [100, 200],           # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.2],    # Learning rate
    'max_depth': [3, 5, 7],               # Maximum tree depth
    'min_samples_split': [2, 5],          # Minimum samples to split
    'subsample': [0.8, 1.0]               # Fraction of samples for training
}

gb_base = GradientBoostingClassifier(random_state=APP_CONFIG['random_state'])

gb_grid = GridSearchCV(
    gb_base,
    gb_param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

print("    Starting grid search (this may take 20-30 minutes)...")
gb_grid.fit(xv_train, y_train)

# Get best model
best_gb = gb_grid.best_estimator_
gb_score = best_gb.score(xv_test, y_test)

results['Gradient Boosting'] = {
    'best_params': gb_grid.best_params_,
    'best_cv_score': gb_grid.best_score_,
    'test_accuracy': gb_score,
    'default_accuracy': GradientBoostingClassifier(
        random_state=APP_CONFIG['random_state'],
        n_estimators=100
    ).fit(xv_train, y_train).score(xv_test, y_test)
}

print(f"\n    [RESULTS]")
print(f"      Best Parameters: {gb_grid.best_params_}")
print(f"      Best CV Score: {gb_grid.best_score_*100:.2f}%")
print(f"      Test Accuracy: {gb_score*100:.2f}%")
print(f"      Improvement over default: {(gb_score - results['Gradient Boosting']['default_accuracy'])*100:+.2f}%")

# Step 5: Save results and models
print("\n[5/5] Saving results...")

# Save tuned models (optional - uncomment if you want to use them)
print("\n  Do you want to save the tuned models? (They will replace current models)")
print("  Type 'yes' to save, or press Enter to skip: ", end='')
# For automated script, we'll save to a separate location
tuned_models_dir = f"{PATHS['models_dir']}tuned/"
import os
os.makedirs(tuned_models_dir, exist_ok=True)

joblib.dump(best_rf, f"{tuned_models_dir}random_forest_tuned.pkl")
joblib.dump(best_gb, f"{tuned_models_dir}gradient_boosting_tuned.pkl")
print(f"  [OK] Saved tuned models to {tuned_models_dir}")

# Save tuning results
tuning_results = {
    'timestamp': datetime.now().isoformat(),
    'cv_folds': 3,
    'total_samples': len(df),
    'training_samples': len(x_train),
    'test_samples': len(x_test),
    'feature_count': xv_train.shape[1],
    'results': {}
}

for model_name, res in results.items():
    tuning_results['results'][model_name] = {
        'best_params': res['best_params'],
        'best_cv_score': float(res['best_cv_score']),
        'test_accuracy': float(res['test_accuracy']),
        'default_accuracy': float(res['default_accuracy']),
        'improvement': float(res['test_accuracy'] - res['default_accuracy'])
    }

output_file = f"{PATHS['models_dir']}hyperparameter_tuning_results.json"
with open(output_file, 'w') as f:
    json.dump(tuning_results, f, indent=2)

print(f"  [OK] Saved results to {output_file}")

# Print summary
print("\n" + "="*70)
print("HYPERPARAMETER TUNING COMPLETE!")
print("="*70)

print("\nSummary:")
print("-"*70)
print(f"{'Model':<25} {'Default':<12} {'Tuned':<12} {'Improvement':<12}")
print("-"*70)

for model_name, res in results.items():
    default = res['default_accuracy'] * 100
    tuned = res['test_accuracy'] * 100
    improvement = (res['test_accuracy'] - res['default_accuracy']) * 100
    print(f"{model_name:<25} {default:<12.2f}% {tuned:<12.2f}% {improvement:+.2f}%")

print("-"*70)

print("\nBest Parameters Found:")
print("-"*70)
for model_name, res in results.items():
    print(f"\n{model_name}:")
    for param, value in res['best_params'].items():
        print(f"  - {param}: {value}")

print("\n" + "="*70)
print(f"\nTuning completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results saved to: {output_file}")
print(f"Tuned models saved to: {tuned_models_dir}")
print("\nNote: To use tuned models, you can replace the default models in the models/ directory")
print("="*70)
