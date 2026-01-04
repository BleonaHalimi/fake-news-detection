"""
Batch Analysis Component
Analyze multiple articles at once
"""

import streamlit as st
import pandas as pd
from utils.prediction import predict_batch, calculate_batch_statistics
from datetime import datetime
import io


def parse_file(uploaded_file):
    """Parse uploaded file and extract articles"""
    articles = []

    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)

            # Try to find text column
            text_columns = ['text', 'article', 'content', 'body', 'description']
            text_col = None

            for col in text_columns:
                if col in df.columns:
                    text_col = col
                    break

            if text_col:
                articles = df[text_col].dropna().tolist()
            else:
                st.error(f"Could not find text column. Available columns: {', '.join(df.columns)}")
                return []

        elif file_extension == 'txt':
            content = uploaded_file.read().decode('utf-8')

            # Split by delimiter
            delimiter = st.session_state.get('delimiter', '\n\n')
            articles = [a.strip() for a in content.split(delimiter) if a.strip()]

        else:
            st.error(f"Unsupported file format: {file_extension}")
            return []

        return articles

    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return []


def render(models):
    """Render the batch analysis page"""

    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #2c3e50;'>Batch Analysis</h1>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Analyze multiple articles at once</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Instructions
    with st.expander("Instructions", expanded=False):
        st.markdown("""
        **Upload Method:**
        1. Prepare a CSV file with a 'text' column containing articles
        2. Or prepare a TXT file with articles separated by double newlines

        **Manual Entry Method:**
        1. Paste multiple articles in the text area
        2. Separate each article with a blank line

        **After upload/entry:**
        1. Review the number of articles detected
        2. Click "Analyze All" to process
        3. View results in table format
        4. Download results as CSV or PDF
        """)

    st.markdown("### Upload Articles")

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Upload File", "Manual Entry"])

    articles = []

    with tab1:
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV or TXT file",
            type=['csv', 'txt'],
            help="CSV must have a 'text' column. TXT files will be split by double newlines."
        )

        if uploaded_file:
            # Delimiter selection for TXT files
            if uploaded_file.name.endswith('.txt'):
                st.session_state['delimiter'] = st.selectbox(
                    "Select delimiter",
                    ['\n\n', '\n---\n', '\n###\n'],
                    format_func=lambda x: {
                        '\n\n': 'Double newline (blank line)',
                        '\n---\n': 'Triple dash (---)',
                        '\n###\n': 'Triple hash (###)'
                    }[x]
                )

            articles = parse_file(uploaded_file)

    with tab2:
        # Manual text entry
        manual_text = st.text_area(
            "Paste articles (separate with blank lines)",
            height=300,
            placeholder="Article 1 text here...\n\nArticle 2 text here...\n\nArticle 3 text here..."
        )

        if manual_text:
            articles = [a.strip() for a in manual_text.split('\n\n') if a.strip()]

    # Display article count
    if articles:
        st.success(f"Found {len(articles)} articles ready for analysis")

        # Show preview
        with st.expander(f"Preview first 3 articles"):
            for idx, article in enumerate(articles[:3], 1):
                st.markdown(f"**Article {idx}:**")
                st.text(article[:200] + '...' if len(article) > 200 else article)
                st.markdown("---")

        st.markdown("<br>", unsafe_allow_html=True)

        # Analyze button
        col1, col2, col3 = st.columns([2, 1, 2])

        with col2:
            if st.button("Analyze All", type="primary", use_container_width=True):
                # Perform batch analysis
                st.markdown("---")

                # Progress tracking
                progress_text = st.empty()
                progress_bar = st.progress(0)
                results_container = st.empty()

                results = []
                vectorizer = models['vectorizer']

                def update_progress(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    progress_text.text(f"Analyzing... {current}/{total} articles ({progress*100:.0f}%)")

                # Run batch prediction
                with st.spinner("Processing articles..."):
                    results = predict_batch(articles, models, vectorizer, update_progress)

                progress_text.empty()
                progress_bar.empty()

                # Calculate statistics
                stats = calculate_batch_statistics(results)

                # Display summary
                st.markdown("### Analysis Summary")

                from components import visualizations
                visualizations.plot_batch_summary(results)

                st.markdown("---")

                # Create results dataframe
                st.markdown("### Detailed Results")

                results_data = []
                for idx, (article, result) in enumerate(zip(articles, results), 1):
                    results_data.append({
                        '#': idx,
                        'Preview': article[:50] + '...' if len(article) > 50 else article,
                        'Consensus': 'TRUE NEWS' if result['consensus'] == 1 else 'FAKE NEWS',
                        'Agreement': f"{result['agreement_count']}/4",
                        'LR': f"{result['model_details'][0]['confidence']:.1f}%",
                        'DT': f"{result['model_details'][1]['confidence']:.1f}%",
                        'GBC': f"{result['model_details'][2]['confidence']:.1f}%",
                        'RFC': f"{result['model_details'][3]['confidence']:.1f}%"
                    })

                df_results = pd.DataFrame(results_data)

                # Display with color coding
                def highlight_consensus(row):
                    if 'TRUE' in str(row['Consensus']):
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)

                st.dataframe(
                    df_results.style.apply(highlight_consensus, axis=1),
                    use_container_width=True,
                    hide_index=True
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Export options
                st.markdown("### Export Results")

                col1, col2 = st.columns(2)

                with col1:
                    # Export as CSV
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        "Download as CSV",
                        csv,
                        f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )

                with col2:
                    # Export detailed results as JSON
                    import json
                    detailed_json = json.dumps(results, indent=2)
                    st.download_button(
                        "Download Detailed JSON",
                        detailed_json,
                        f"batch_analysis_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True
                    )

    else:
        st.info("Upload a file or paste articles above to begin analysis")
