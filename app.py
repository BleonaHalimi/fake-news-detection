"""
Fake News Detector - Streamlit Web Application
Main application entry point
"""

import streamlit as st
from config import APP_CONFIG, NAV_PAGES, CUSTOM_CSS
from utils.model_manager import load_models, get_model_accuracies, check_models_exist

# Page configuration
st.set_page_config(
    page_title=APP_CONFIG['page_title'],
    page_icon=APP_CONFIG['page_icon'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

if 'current_page' not in st.session_state:
    st.session_state.current_page = NAV_PAGES['home']

# Check if models exist
models_exist, missing_files = check_models_exist()

if not models_exist:
    st.error("**Error: Models not found!**")
    st.write("Please train the models first by running:")
    st.code("python train_models.py", language="bash")
    st.write("**Missing files:**")
    for file in missing_files:
        st.write(f"  - {file}")
    st.stop()

# Load models (cached)
try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Load model accuracies
accuracies = get_model_accuracies()

# Sidebar
with st.sidebar:
    # Title
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #3498db; margin: 0;'>Fake News</h1>
        <h1 style='color: #3498db; margin: 0;'>Detection System</h1>
        <p style='color: #7f8c8d; margin-top: 10px;'>Machine Learning Verification</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Navigation
    selected_page = st.radio(
        "**Navigation**",
        list(NAV_PAGES.values()),
        index=0,
        label_visibility="visible"
    )

    st.divider()

    # Model Performance Section
    st.subheader("Model Performance")

    if accuracies:
        for model_name, accuracy in accuracies.items():
            # Color code based on accuracy
            if accuracy >= 99:
                color = "#27ae60"  # Green
            elif accuracy >= 95:
                color = "#3498db"  # Blue
            else:
                color = "#f39c12"  # Orange

            st.markdown(f"""
            <div style='background: {color}; padding: 10px; border-radius: 5px; margin: 5px 0;'>
                <p style='color: white; margin: 0; font-size: 0.9em;'><strong>{model_name}</strong></p>
                <p style='color: white; margin: 0; font-size: 1.2em;'>{accuracy}%</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Model accuracy data not available")

    st.divider()

    # Project Attribution
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 10px; font-size: 0.75em;'>
        <p style='margin: 0;'><strong>University Project</strong></p>
        <p style='margin: 5px 0 0 0;'>Uestli Guci<br>Bleona Halimi<br>Aurora Destani<br>Petrit Qerimi</p>
    </div>
    """, unsafe_allow_html=True)

# Main content area
if selected_page == NAV_PAGES['home']:
    # Home page
    st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='color: #2c3e50; font-size: 3em;'>Fake News Detection System</h1>
        <p style='color: #7f8c8d; font-size: 1.3em;'>Verify news authenticity with AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #3498db;'>Single Analysis</h3>
            <p>Paste any article and get instant verification from 4 different AI models</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #9b59b6;'>Batch Processing</h3>
            <p>Upload multiple articles at once for bulk analysis and export results</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3 style='color: #e74c3c;'>URL Analysis</h3>
            <p>Extract and analyze articles directly from web URLs automatically</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # How it works
    st.markdown("### System Methodology")

    st.markdown("""
    Our system uses **4 powerful machine learning models** working together:

    1. **Logistic Regression** - Fast and accurate linear classification
    2. **Decision Tree** - Pattern-based analysis
    3. **Gradient Boosting** - Advanced ensemble learning
    4. **Random Forest** - Robust multi-tree classifier

    Each model independently analyzes the article, and we show you all 4 predictions plus a **consensus result** for maximum accuracy.
    """)

    st.markdown("---")

    # Statistics
    st.markdown("### System Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Models Trained", "4", help="Four different ML algorithms")

    with col2:
        avg_acc = sum(accuracies.values()) / len(accuracies) if accuracies else 99
        st.metric("Avg Accuracy", f"{avg_acc:.1f}%", help="Average across all models")

    with col3:
        st.metric("Training Data", "44,878", help="Total articles used for training")

    with col4:
        st.metric("Features", "5,000", help="TF-IDF features extracted")

    st.markdown("---")

    # Get started
    st.info("Select an analysis option from the sidebar to begin.")

elif selected_page == NAV_PAGES['single']:
    from components import single_analysis
    single_analysis.render(models)

elif selected_page == NAV_PAGES['batch']:
    from components import batch_analysis
    batch_analysis.render(models)

elif selected_page == NAV_PAGES['url']:
    from components import url_analysis
    url_analysis.render(models)

elif selected_page == NAV_PAGES['history']:
    from components import history_viewer
    history_viewer.render(models)
