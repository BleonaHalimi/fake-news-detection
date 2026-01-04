"""
Prediction Utilities
Handles predictions for single and batch articles
"""

from utils.text_preprocessing import wordopt, extract_advanced_features, features_to_array
import pandas as pd
import numpy as np
from scipy.sparse import hstack


def predict_single(text, models, vectorizer, accuracies=None, use_weighted=True):
    """
    Predict whether a single article is fake or true using all 4 models

    Args:
        text (str): Article text to analyze
        models (dict): Dictionary of trained models
        vectorizer: Trained TF-IDF vectorizer
        accuracies (dict): Optional dictionary of model accuracies for weighted voting
        use_weighted (bool): Use weighted voting if accuracies provided

    Returns:
        dict: Prediction results
            {
                'text': original text,
                'text_preview': truncated text,
                'predictions': [0/1 for each model],
                'confidences': [confidence % for each model],
                'consensus': 0 or 1,
                'agreement_count': number of models that agree,
                'weighted_confidence': confidence from weighted voting,
                'model_details': [
                    {
                        'name': model name,
                        'prediction': 0 or 1,
                        'confidence': percentage,
                        'label': 'FAKE NEWS' or 'TRUE NEWS'
                    },
                    ...
                ]
            }
    """
    # Preprocess text for TF-IDF
    cleaned_text = wordopt(text)

    # Vectorize
    tfidf_features = vectorizer.transform([cleaned_text])

    # Extract advanced features
    try:
        advanced_feats = extract_advanced_features(text)
        advanced_array = features_to_array(advanced_feats).reshape(1, -1)
        # Combine TF-IDF with advanced features
        vectorized = hstack([tfidf_features, advanced_array])
    except Exception as e:
        # Fallback to TF-IDF only if advanced features fail
        vectorized = tfidf_features

    # Get predictions and probabilities from all models
    predictions = []
    confidences = []
    model_details = []
    probabilities = []

    model_keys = ['lr', 'dt', 'gbc', 'rfc']
    model_names = ['Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'Random Forest']

    for model_key, model_name in zip(model_keys, model_names):
        model = models[model_key]

        # Get prediction
        pred = int(model.predict(vectorized)[0])
        predictions.append(pred)

        # Get confidence (probability of the predicted class)
        try:
            proba = model.predict_proba(vectorized)[0]
            probabilities.append(proba)
            confidence = float(proba[pred] * 100)
        except:
            # Some models might not have predict_proba
            proba = np.array([1.0, 0.0] if pred == 0 else [0.0, 1.0])
            probabilities.append(proba)
            confidence = 100.0 if pred == 1 else 0.0

        confidences.append(confidence)

        # Create model detail
        model_details.append({
            'name': model_name,
            'prediction': pred,
            'confidence': round(confidence, 1),
            'label': 'TRUE NEWS' if pred == 1 else 'FAKE NEWS'
        })

    # Calculate consensus using weighted voting if accuracies provided
    if use_weighted and accuracies:
        weighted_vote = 0.0
        total_weight = 0.0

        for model_key, model_name in zip(model_keys, model_names):
            if model_name in accuracies:
                weight = accuracies[model_name] / 100.0  # Normalize accuracy as weight
                # Use probability of class 1 (TRUE NEWS)
                model_idx = model_keys.index(model_key)
                weighted_vote += probabilities[model_idx][1] * weight
                total_weight += weight

        # Weighted average probability of TRUE NEWS
        weighted_prob = weighted_vote / total_weight if total_weight > 0 else 0.5
        consensus = 1 if weighted_prob >= 0.5 else 0
        weighted_confidence = weighted_prob * 100 if consensus == 1 else (1 - weighted_prob) * 100
    else:
        # Simple majority vote
        consensus = 1 if sum(predictions) >= 2 else 0  # At least 2 out of 4 models
        weighted_confidence = None

    # Count agreement
    agreement_count = sum(1 for p in predictions if p == consensus)

    result = {
        'text': text,
        'text_preview': text[:100] + '...' if len(text) > 100 else text,
        'predictions': predictions,
        'confidences': confidences,
        'consensus': consensus,
        'agreement_count': agreement_count,
        'model_details': model_details
    }

    if weighted_confidence is not None:
        result['weighted_confidence'] = round(weighted_confidence, 1)
        result['voting_method'] = 'weighted'
    else:
        result['voting_method'] = 'majority'

    return result


def predict_batch(texts, models, vectorizer, progress_callback=None):
    """
    Predict for multiple articles

    Args:
        texts (list): List of article texts
        models (dict): Dictionary of trained models
        vectorizer: Trained TF-IDF vectorizer
        progress_callback (callable): Optional callback function to report progress
            Should accept two args: (current_index, total_count)

    Returns:
        list: List of prediction results (same format as predict_single)
    """
    results = []

    for idx, text in enumerate(texts):
        result = predict_single(text, models, vectorizer)
        results.append(result)

        # Call progress callback if provided
        if progress_callback:
            progress_callback(idx + 1, len(texts))

    return results


def get_prediction_summary(result):
    """
    Get a formatted summary of a prediction result

    Args:
        result (dict): Prediction result from predict_single

    Returns:
        str: Formatted summary
    """
    consensus_label = 'TRUE NEWS' if result['consensus'] == 1 else 'FAKE NEWS'
    agreement = result['agreement_count']

    summary = f"Consensus: {consensus_label} ({agreement}/4 models agree)\n\n"

    for detail in result['model_details']:
        summary += f"{detail['name']}: {detail['label']} ({detail['confidence']:.1f}% confidence)\n"

    return summary


def calculate_batch_statistics(results):
    """
    Calculate statistics for batch predictions

    Args:
        results (list): List of prediction results

    Returns:
        dict: Statistics
            {
                'total': total count,
                'fake_count': count of fake news,
                'true_count': count of true news,
                'avg_confidence': average confidence,
                'high_confidence_count': count with >90% confidence,
                'unanimous_count': count where all 4 models agree
            }
    """
    if not results:
        return {}

    fake_count = sum(1 for r in results if r['consensus'] == 0)
    true_count = sum(1 for r in results if r['consensus'] == 1)

    # Calculate average confidence
    all_confidences = []
    for r in results:
        all_confidences.extend(r['confidences'])
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

    # Count high confidence predictions (>90%)
    high_confidence_count = sum(1 for r in results
                                for conf in r['confidences']
                                if conf > 90)

    # Count unanimous predictions (all 4 models agree)
    unanimous_count = sum(1 for r in results if r['agreement_count'] == 4)

    return {
        'total': len(results),
        'fake_count': fake_count,
        'true_count': true_count,
        'avg_confidence': round(avg_confidence, 1),
        'high_confidence_count': high_confidence_count,
        'unanimous_count': unanimous_count
    }
