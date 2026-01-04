"""
Single Article Analysis Component
Beautiful UI for analyzing individual news articles
"""

import streamlit as st
from utils.prediction import predict_single, get_prediction_summary
from utils.text_preprocessing import validate_text, truncate_text, count_words
from config import COLORS, APP_CONFIG
import json
import os
from datetime import datetime
import uuid


def save_to_history(result):
    """Save analysis result to history"""
    history_file = 'history/analysis_history.json'

    # Create history directory if it doesn't exist
    os.makedirs('history', exist_ok=True)

    # Load existing history
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            try:
                history = json.load(f)
            except:
                history = []
    else:
        history = []

    # Add new entry
    entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'text_preview': result['text_preview'],
        'consensus': result['consensus'],
        'agreement_count': result['agreement_count'],
        'model_details': result['model_details'],
        'source': 'manual_input'
    }

    history.append(entry)

    # Keep only last 100 entries
    history = history[-APP_CONFIG['max_history_entries']:]

    # Save
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    # Also add to session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(entry)


def render(models):
    """Render the single article analysis page"""

    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2c3e50;'>Single Article Analysis</h1>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Analyze any news article for authenticity</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Instructions
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        1. **Paste or type** a news article in the text area below
        2. Make sure the article is at least **50 characters** long
        3. Click the **"Analyze Article"** button
        4. View predictions from all 4 AI models
        5. See the **consensus result** for final verdict

        **Tip:** The longer and more complete the article, the more accurate the analysis!
        """)

    # Text input
    article_text = st.text_area(
        "**Enter news article to analyze:**",
        height=300,
        placeholder="Paste your news article here...\n\nExample: 'WASHINGTON (Reuters) - The United States government announced new economic policies today...'",
        help="Minimum 50 characters required for analysis"
    )

    # Character and word count
    char_count = len(article_text.strip())
    word_count = count_words(article_text.strip()) if article_text.strip() else 0

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if char_count == 0:
            st.info("Enter article text above to begin analysis.")
        elif char_count < APP_CONFIG['min_text_length']:
            remaining = APP_CONFIG['min_text_length'] - char_count
            st.warning(f"Minimum length not met. Need {remaining} more characters (currently {char_count} characters)")
        else:
            st.success(f"{char_count} characters - Ready for analysis")

    with col2:
        st.metric("Characters", f"{char_count:,}")

    with col3:
        st.metric("Words", f"{word_count:,}")

    # Analyze button
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        analyze_button = st.button(
            "Analyze Article",
            type="primary",
            use_container_width=True,
            disabled=(char_count < APP_CONFIG['min_text_length'])
        )

    # Perform analysis
    if analyze_button:
        # Validate text
        is_valid, error_msg = validate_text(article_text, APP_CONFIG['min_text_length'])

        if not is_valid:
            st.error(f"Error: {error_msg}")
            st.stop()

        # Show spinner while analyzing
        with st.spinner("Analyzing article with ensemble of 4 classifiers..."):
            from utils.model_manager import get_model_accuracies
            vectorizer = models['vectorizer']
            accuracies = get_model_accuracies()
            result = predict_single(article_text, models, vectorizer, accuracies=accuracies, use_weighted=True)

        st.markdown("---")
        st.markdown("<br>", unsafe_allow_html=True)

        # Display consensus banner
        consensus = result['consensus']
        agreement = result['agreement_count']

        if consensus == 1:
            st.markdown(f"""
            <div class='consensus-true'>
                <div style='font-size: 2em; margin-bottom: 10px;'>CONSENSUS: VERIFIED AS TRUE NEWS</div>
                <div style='font-size: 1.2em; opacity: 0.9;'>{agreement} out of 4 models agree</div>
                <div style='font-size: 0.9em; margin-top: 10px; opacity: 0.8;'>Confidence: {(agreement/4)*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='consensus-fake'>
                <div style='font-size: 2em; margin-bottom: 10px;'>CONSENSUS: DETECTED AS FAKE NEWS</div>
                <div style='font-size: 1.2em; opacity: 0.9;'>{agreement} out of 4 models agree</div>
                <div style='font-size: 0.9em; margin-top: 10px; opacity: 0.8;'>Confidence: {(agreement/4)*100:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Display individual model results
        st.markdown("### Individual Classifier Predictions")

        cols = st.columns(4)

        for idx, (col, detail) in enumerate(zip(cols, result['model_details'])):
            with col:
                pred = detail['prediction']
                conf = detail['confidence']
                model_name = detail['name']
                label = detail['label']

                # Determine card class based on prediction
                if pred == 1:
                    card_html = f"""
                    <div class='result-card-true'>
                        <div class='model-header'>{model_name}</div>
                        <div class='model-prediction'>TRUE</div>
                        <div class='confidence-text'>{conf:.1f}%</div>
                    </div>
                    """
                else:
                    card_html = f"""
                    <div class='result-card-fake'>
                        <div class='model-header'>{model_name}</div>
                        <div class='model-prediction'>FAKE</div>
                        <div class='confidence-text'>{conf:.1f}%</div>
                    </div>
                    """

                st.markdown(card_html, unsafe_allow_html=True)

                # Progress bar
                st.progress(conf / 100.0)

        st.markdown("<br>", unsafe_allow_html=True)

        # Enhanced visualizations with tabs
        st.markdown("### Detailed Analysis")

        from components import visualizations

        tab1, tab2, tab3 = st.tabs([
            "Model Predictions",
            "LIME Explanation",
            "Feature Importance"
        ])

        with tab1:
            st.markdown("#### Confidence Scores")
            visualizations.plot_confidence_chart(result)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Model Agreement")
            visualizations.plot_model_agreement(result)

        with tab2:
            st.markdown("""
            <div style='background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <h4 style='margin: 0; color: #1e3a8a;'>LIME Methodology</h4>
                <p style='margin: 10px 0 0 0; color: #374151;'>
                    LIME (Local Interpretable Model-agnostic Explanations) shows which specific words
                    in this article influenced the prediction. This helps you understand <strong>why</strong>
                    the model made its decision.
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Generating LIME explanation... This may take 10-20 seconds..."):
                try:
                    vectorizer = models['vectorizer']
                    explanation = visualizations.explain_prediction_with_lime(
                        article_text,
                        models['rfc'],  # Use Random Forest (best model)
                        vectorizer
                    )
                    visualizations.visualize_lime_explanation(explanation, result['consensus'])
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {str(e)}")
                    st.info("LIME analysis requires the article text and trained models. Please try again.")

        with tab3:
            st.markdown("""
            <div style='background-color: #f0fdf4; padding: 15px; border-radius: 8px; margin-bottom: 20px;'>
                <h4 style='margin: 0; color: #065f46;'>Feature Importance Analysis</h4>
                <p style='margin: 10px 0 0 0; color: #374151;'>
                    This shows the most important words that the Random Forest model uses
                    to distinguish fake from true news across <strong>all articles</strong>,
                    not just this one.
                </p>
            </div>
            """, unsafe_allow_html=True)

            try:
                vectorizer = models['vectorizer']
                feature_names = vectorizer.get_feature_names_out()
                visualizations.plot_feature_importance(
                    models['rfc'],  # Use Random Forest
                    feature_names,
                    top_n=20
                )
            except Exception as e:
                st.error(f"Error generating feature importance: {str(e)}")
                st.info("Feature importance analysis requires trained models.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Save to History", use_container_width=True):
                save_to_history(result)
                st.success("Saved to analysis history")

        with col2:
            # Download as text
            summary = get_prediction_summary(result)
            st.download_button(
                "Download Summary",
                summary,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                use_container_width=True
            )

        with col3:
            if st.button("Analyze Another Article", use_container_width=True):
                st.rerun()

        st.markdown("---")

        # Detailed breakdown (collapsible)
        with st.expander("Detailed Analysis Breakdown"):
            st.markdown("#### Article Preview")
            st.text_area("", result['text_preview'], height=100, disabled=True)

            st.markdown("#### Model Predictions")
            import pandas as pd
            df = pd.DataFrame(result['model_details'])
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("#### Interpretation")
            if agreement == 4:
                st.success("**Unanimous Decision:** All 4 models agree on this prediction. Very high confidence!")
            elif agreement == 3:
                st.info("**Strong Consensus:** 3 out of 4 models agree. High confidence in this prediction.")
            elif agreement == 2:
                st.warning("**Split Decision:** Models are evenly divided. Consider additional verification.")
            else:
                st.error("**No Consensus:** Models strongly disagree. This article may need manual review.")
