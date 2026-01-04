"""
Text Preprocessing Utilities
Contains functions for cleaning and validating text input
"""

import re
import string


def wordopt(text):
    """
    Clean and preprocess text for fake news detection

    Args:
        text (str): Raw text input

    Returns:
        str: Cleaned and preprocessed text
    """
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)  # Remove text in brackets
    text = re.sub(r"\\W", " ", text)      # Replace non-word characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)    # Remove HTML tags
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)        # Remove newlines
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    return text


def validate_text(text, min_length=50):
    """
    Validate text input for minimum length and content

    Args:
        text (str): Text to validate
        min_length (int): Minimum required character count

    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze"

    if len(text.strip()) < min_length:
        return False, f"Text must be at least {min_length} characters long (currently {len(text.strip())} characters)"

    # Check if text contains at least some alphabetic characters
    if not re.search(r'[a-zA-Z]', text):
        return False, "Text must contain alphabetic characters"

    return True, None


def clean_html(text):
    """
    Remove HTML tags and entities from text

    Args:
        text (str): Text potentially containing HTML

    Returns:
        str: Text with HTML removed
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', '', text)

    return text


def truncate_text(text, max_length=100, suffix='...'):
    """
    Truncate text to a maximum length

    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        suffix (str): Suffix to add if truncated

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)].strip() + suffix


def count_words(text):
    """
    Count number of words in text

    Args:
        text (str): Input text

    Returns:
        int: Word count
    """
    return len(text.split())


def extract_advanced_features(text):
    """
    Extract advanced linguistic and statistical features from text

    These features help distinguish fake from true news based on:
    - Sentiment patterns (fake news often more extreme/emotional)
    - Readability (fake news often simpler language)
    - Text statistics (punctuation, capitalization patterns)

    Args:
        text (str): Original text (before preprocessing)

    Returns:
        dict: Dictionary with 8 numerical features
    """
    from textblob import TextBlob
    import textstat
    import numpy as np

    # Handle empty text
    if not text or not text.strip():
        return {
            'sentiment_polarity': 0.0,
            'sentiment_subjectivity': 0.0,
            'flesch_reading_ease': 0.0,
            'avg_sentence_length': 0.0,
            'word_count': 0,
            'unique_word_ratio': 0.0,
            'punctuation_ratio': 0.0,
            'capital_ratio': 0.0
        }

    features = {}

    # 1. Sentiment Analysis (TextBlob)
    try:
        blob = TextBlob(text)
        features['sentiment_polarity'] = blob.sentiment.polarity  # -1 to 1
        features['sentiment_subjectivity'] = blob.sentiment.subjectivity  # 0 to 1
    except:
        features['sentiment_polarity'] = 0.0
        features['sentiment_subjectivity'] = 0.0

    # 2. Readability Metrics
    try:
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['avg_sentence_length'] = textstat.avg_sentence_length(text)
    except:
        features['flesch_reading_ease'] = 0.0
        features['avg_sentence_length'] = 0.0

    # 3. Text Statistics
    words = text.split()
    features['word_count'] = len(words)

    # Unique word ratio (vocabulary diversity)
    if len(words) > 0:
        features['unique_word_ratio'] = len(set(words)) / len(words)
    else:
        features['unique_word_ratio'] = 0.0

    # Punctuation ratio (excessive punctuation can indicate sensationalism)
    if len(text) > 0:
        punct_count = sum(1 for char in text if char in string.punctuation)
        features['punctuation_ratio'] = punct_count / len(text)

        # Capital letter ratio (ALL CAPS for emphasis is common in fake news)
        capital_count = sum(1 for char in text if char.isupper())
        features['capital_ratio'] = capital_count / len(text)
    else:
        features['punctuation_ratio'] = 0.0
        features['capital_ratio'] = 0.0

    return features


def features_to_array(features):
    """
    Convert feature dictionary to ordered numpy array

    Args:
        features (dict): Feature dictionary from extract_advanced_features()

    Returns:
        numpy.array: 1D array of 8 features in consistent order
    """
    import numpy as np

    return np.array([
        features['sentiment_polarity'],
        features['sentiment_subjectivity'],
        features['flesch_reading_ease'],
        features['avg_sentence_length'],
        features['word_count'],
        features['unique_word_ratio'],
        features['punctuation_ratio'],
        features['capital_ratio']
    ])
