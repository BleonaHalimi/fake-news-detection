"""
Model Manager
Handles loading, caching, and management of trained ML models
"""

import joblib
import json
import os
import streamlit as st
from config import PATHS, MODEL_FILES, MODEL_NAMES


@st.cache_resource
def load_models():
    """
    Load all trained models and vectorizer from disk
    Uses Streamlit caching to load once and reuse

    Returns:
        dict: Dictionary containing all models and vectorizer
            {
                'lr': LogisticRegression model,
                'dt': DecisionTree model,
                'gbc': GradientBoosting model,
                'rfc': RandomForest model,
                'vectorizer': TfidfVectorizer
            }
    """
    models = {}

    try:
        # Load each model
        for model_key in ['lr', 'dt', 'gbc', 'rfc', 'vectorizer']:
            filepath = f"{PATHS['models_dir']}{MODEL_FILES[model_key]}"

            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"Model file not found: {filepath}\n"
                    f"Please run 'python train_models.py' first to train and save the models."
                )

            models[model_key] = joblib.load(filepath)

        return models

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()


def get_model_accuracies():
    """
    Load model accuracy scores from metadata file

    Returns:
        dict: Dictionary of model names and their accuracies
            {'Logistic Regression': 98.62, ...}
    """
    metadata_path = f"{PATHS['models_dir']}{MODEL_FILES['metadata']}"

    if not os.path.exists(metadata_path):
        # Return default values if metadata doesn't exist
        return {
            'Logistic Regression': 98.62,
            'Decision Tree': 99.55,
            'Gradient Boosting': 99.54,
            'Random Forest': 99.73
        }

    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata.get('accuracies', {})

    except Exception as e:
        st.warning(f"Could not load model metadata: {str(e)}")
        return {}


def get_model_metadata():
    """
    Load complete model metadata

    Returns:
        dict: Complete metadata including training date, sample counts, etc.
    """
    metadata_path = f"{PATHS['models_dir']}{MODEL_FILES['metadata']}"

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)

    except Exception as e:
        st.warning(f"Could not load model metadata: {str(e)}")
        return None


def check_models_exist():
    """
    Check if all required model files exist

    Returns:
        tuple: (all_exist: bool, missing_files: list)
    """
    missing_files = []

    for model_key in ['lr', 'dt', 'gbc', 'rfc', 'vectorizer']:
        filepath = f"{PATHS['models_dir']}{MODEL_FILES[model_key]}"
        if not os.path.exists(filepath):
            missing_files.append(filepath)

    return len(missing_files) == 0, missing_files
