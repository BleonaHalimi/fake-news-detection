"""
Configuration file for Fake News Detection Application
Contains color schemes, settings, and constants
"""

# Color Scheme
COLORS = {
    # Results
    'true_news': '#27ae60',      # Vibrant green
    'true_light': '#2ecc71',     # Light green
    'fake_news': '#c0392b',      # Dark red
    'fake_light': '#e74c3c',     # Light red
    'neutral': '#95a5a6',        # Gray

    # UI Elements
    'primary': '#3498db',        # Blue
    'secondary': '#9b59b6',      # Purple
    'warning': '#f39c12',        # Orange
    'info': '#16a085',           # Teal

    # Backgrounds
    'bg_light': '#ecf0f1',       # Light background
    'bg_dark': '#34495e',        # Dark background
    'card_bg': '#ffffff',        # Card background

    # Text
    'text_primary': '#2c3e50',   # Dark text
    'text_secondary': '#7f8c8d', # Gray text
    'text_light': '#ffffff',     # White text
}

# Model names mapping
MODEL_NAMES = {
    'lr': 'Logistic Regression',
    'dt': 'Decision Tree',
    'gbc': 'Gradient Boosting',
    'rfc': 'Random Forest'
}

# File paths
PATHS = {
    'models_dir': 'models/',
    'data_dir': 'data/',
    'history_dir': 'history/',
    'exports_dir': 'exports/',
    'assets_dir': 'assets/'
}

# Model file names
MODEL_FILES = {
    'lr': 'logistic_regression.pkl',
    'dt': 'decision_tree.pkl',
    'gbc': 'gradient_boosting.pkl',
    'rfc': 'random_forest.pkl',
    'vectorizer': 'vectorizer.pkl',
    'metadata': 'model_metadata.json'
}

# Application settings
APP_CONFIG = {
    'page_title': 'Fake News Detection System',
    'page_icon': '📰',
    'layout': 'wide',
    'min_text_length': 50,
    'max_history_entries': 100,
    'tfidf_max_features': 5000,
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5
}

# Navigation pages
NAV_PAGES = {
    'home': 'Home',
    'single': 'Single Article Analysis',
    'batch': 'Batch Analysis',
    'history': 'Analysis History'
}

# CSS Styles
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }

    /* Result card styling - True News */
    .result-card-true {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e063 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin: 10px 0;
        text-align: center;
    }

    /* Result card styling - Fake News */
    .result-card-fake {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        margin: 10px 0;
        text-align: center;
    }

    /* Consensus banner - True */
    .consensus-true {
        background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
        text-align: center;
        margin: 20px 0;
        font-size: 1.5em;
        font-weight: bold;
    }

    /* Consensus banner - Fake */
    .consensus-fake {
        background: linear-gradient(90deg, #c0392b 0%, #e74c3c 100%);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
        text-align: center;
        margin: 20px 0;
        font-size: 1.5em;
        font-weight: bold;
    }

    /* Model card header */
    .model-header {
        font-size: 0.9em;
        font-weight: 600;
        margin-bottom: 10px;
        opacity: 0.9;
    }

    /* Model prediction text */
    .model-prediction {
        font-size: 1.3em;
        font-weight: bold;
        margin: 10px 0;
    }

    /* Confidence text */
    .confidence-text {
        font-size: 1.1em;
        margin-top: 10px;
    }

    /* Info box */
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 10px 0;
    }

    /* Warning box */
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f39c12;
        margin: 10px 0;
    }

    /* Success box */
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #27ae60;
        margin: 10px 0;
    }

    /* Error box */
    .error-box {
        background: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #e74c3c;
        margin: 10px 0;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Button styling */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
"""
