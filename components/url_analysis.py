"""
URL Analysis Component
Extract and analyze articles from URLs
"""

import streamlit as st
from utils.web_scraper import extract_article_from_url, is_valid_url
from utils.prediction import predict_single
from utils.text_preprocessing import validate_text, count_words
from config import APP_CONFIG


def render(models):
    """Render the URL analysis page"""

    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2c3e50;'>URL Analysis</h1>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Extract and analyze articles from web URLs</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Instructions
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        1. **Copy** the URL of a news article
        2. **Paste** it in the input field below
        3. Click **"Fetch Article"** to extract the content
        4. **Review** the extracted text (you can edit if needed)
        5. Click **"Analyze"** to get predictions

        **Supported Sources:**
        - Most news websites (Reuters, CNN, BBC, etc.)
        - Blog posts
        - Online articles

        **Note:** Some websites may block automated scraping. If extraction fails, you can copy-paste the text manually in the Single Analysis page.
        """)

    st.markdown("### Enter Article URL")

    # URL input
    url_input = st.text_input(
        "**Article URL:**",
        placeholder="https://example.com/news-article",
        help="Enter the full URL including http:// or https://"
    )

    # Validate URL
    if url_input and not is_valid_url(url_input):
        st.warning("Please enter a valid URL starting with http:// or https://")

    # Fetch button
    col1, col2, col3 = st.columns([2, 1, 2])

    with col2:
        fetch_button = st.button(
            "Fetch Article",
            type="primary",
            use_container_width=True,
            disabled=not url_input or not is_valid_url(url_input)
        )

    # Fetch and display article
    if fetch_button:
        with st.spinner("Fetching article from URL..."):
            article_data, error = extract_article_from_url(url_input)

        if error:
            st.error(f"Error: {error}")
            st.info("""
            **Troubleshooting:**
            - Make sure the URL is correct and accessible
            - Some websites block automated access
            - Try using the "Single Analysis" page and paste the text manually
            """)
        else:
            st.success("Article extracted successfully")
            st.markdown("---")

            # Display extracted information
            st.markdown("### Extracted Article")

            if article_data.get('title'):
                st.markdown(f"**Title:** {article_data['title']}")

            st.markdown(f"**Source:** {url_input}")

            # Show article text in editable text area
            st.markdown("**Article Text:** (you can edit if needed)")
            extracted_text = st.text_area(
                "Article content",
                value=article_data['text'],
                height=300,
                label_visibility="collapsed"
            )

            # Character and word count
            char_count = len(extracted_text.strip())
            word_count = count_words(extracted_text.strip())

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Characters", f"{char_count:,}")
            with col2:
                st.metric("Words", f"{word_count:,}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Analyze button
            col1, col2, col3 = st.columns([2, 1, 2])

            with col2:
                analyze_button = st.button(
                    "Analyze Article",
                    type="primary",
                    use_container_width=True,
                    disabled=(char_count < APP_CONFIG['min_text_length'])
                )

            if analyze_button:
                # Validate text
                is_valid, error_msg = validate_text(extracted_text, APP_CONFIG['min_text_length'])

                if not is_valid:
                    st.error(f"Error: {error_msg}")
                else:
                    # Perform analysis
                    with st.spinner("Analyzing article with ensemble of 4 classifiers..."):
                        vectorizer = models['vectorizer']
                        result = predict_single(extracted_text, models, vectorizer)

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
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class='consensus-fake'>
                            <div style='font-size: 2em; margin-bottom: 10px;'>CONSENSUS: DETECTED AS FAKE NEWS</div>
                            <div style='font-size: 1.2em; opacity: 0.9;'>{agreement} out of 4 models agree</div>
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
                            st.progress(conf / 100.0)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Visualizations
                    st.markdown("### Confidence Scores")

                    from components import visualizations
                    visualizations.plot_confidence_chart(result)

    else:
        # Show example URLs
        st.markdown("### Example URLs")

        example_urls = [
            "https://www.reuters.com/world/...",
            "https://www.bbc.com/news/...",
            "https://www.theguardian.com/...",
            "https://apnews.com/..."
        ]

        st.info("""
        **Try pasting URLs from major news sources like:**
        - Reuters
        - BBC News
        - Associated Press (AP)
        - The Guardian
        - CNN
        - And many others!
        """)
