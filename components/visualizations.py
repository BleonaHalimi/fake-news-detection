"""
Visualizations Component
Creates charts and graphs for analysis results
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from config import COLORS
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer


def plot_confidence_chart(result):
    """
    Plot confidence scores for all models as a horizontal bar chart

    Args:
        result (dict): Prediction result from predict_single
    """
    model_details = result['model_details']

    # Extract data
    model_names = [d['name'] for d in model_details]
    confidences = [d['confidence'] for d in model_details]
    predictions = [d['prediction'] for d in model_details]

    # Create colors based on prediction
    colors = [COLORS['true_news'] if p == 1 else COLORS['fake_news'] for p in predictions]

    # Create horizontal bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=model_names,
        x=confidences,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        text=[f"{c:.1f}%" for c in confidences],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        title="Model Confidence Scores",
        xaxis_title="Confidence (%)",
        yaxis_title="",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(range=[0, 100], gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_model_agreement(result):
    """
    Plot a pie chart showing model agreement

    Args:
        result (dict): Prediction result
    """
    agree_count = result['agreement_count']
    disagree_count = 4 - agree_count

    fig = go.Figure(data=[go.Pie(
        labels=['Agree', 'Disagree'],
        values=[agree_count, disagree_count],
        marker=dict(colors=[COLORS['primary'], COLORS['neutral']]),
        hole=0.4,
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title=f"Model Agreement ({agree_count}/4 models)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_batch_summary(results):
    """
    Plot summary statistics for batch analysis

    Args:
        results (list): List of prediction results
    """
    from utils.prediction import calculate_batch_statistics

    stats = calculate_batch_statistics(results)

    # Create bar chart for fake vs true
    fig = go.Figure(data=[
        go.Bar(
            x=['Fake News', 'True News'],
            y=[stats['fake_count'], stats['true_count']],
            marker=dict(color=[COLORS['fake_news'], COLORS['true_news']]),
            text=[stats['fake_count'], stats['true_count']],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
    ])

    fig.update_layout(
        title=f"Batch Analysis Results (Total: {stats['total']} articles)",
        xaxis_title="",
        yaxis_title="Number of Articles",
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Articles", stats['total'])

    with col2:
        st.metric("Average Confidence", f"{stats['avg_confidence']:.1f}%")

    with col3:
        st.metric("Unanimous Decisions", stats['unanimous_count'])

    with col4:
        fake_percentage = (stats['fake_count'] / stats['total'] * 100) if stats['total'] > 0 else 0
        st.metric("Fake News %", f"{fake_percentage:.1f}%")


def plot_gauge(value, title="Confidence", max_value=100):
    """
    Plot a gauge chart for a single value

    Args:
        value (float): Value to display
        title (str): Chart title
        max_value (float): Maximum value for the gauge
    """
    # Determine color based on value
    if value >= 75:
        color = COLORS['true_news']
    elif value >= 50:
        color = COLORS['warning']
    else:
        color = COLORS['fake_news']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_history_timeline(history):
    """
    Plot timeline of analysis history

    Args:
        history (list): List of history entries
    """
    if not history:
        st.info("No history data available")
        return

    import pandas as pd
    from datetime import datetime

    # Prepare data
    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['result'] = df['consensus'].apply(lambda x: 'True' if x == 1 else 'Fake')

    # Count by date and result
    timeline = df.groupby(['date', 'result']).size().reset_index(name='count')

    # Create stacked bar chart
    fig = px.bar(
        timeline,
        x='date',
        y='count',
        color='result',
        title="Analysis History Timeline",
        labels={'date': 'Date', 'count': 'Number of Articles', 'result': 'Result'},
        color_discrete_map={'True': COLORS['true_news'], 'Fake': COLORS['fake_news']},
        barmode='stack'
    )

    fig.update_layout(
        height=350,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)


def explain_prediction_with_lime(text, model, vectorizer, class_names=['Fake News', 'True News']):
    """
    Generate LIME explanation for a prediction

    Args:
        text: Original article text
        model: Trained model to explain (usually Random Forest - best performing)
        vectorizer: TF-IDF vectorizer
        class_names: Names of classes

    Returns:
        LIME explanation object
    """
    from utils.text_preprocessing import wordopt, extract_advanced_features, features_to_array
    from scipy.sparse import hstack

    # Create prediction function for LIME
    def predict_proba_fn(texts):
        # Preprocess texts for TF-IDF
        processed = [wordopt(t) for t in texts]
        # Vectorize with TF-IDF
        tfidf_vectors = vectorizer.transform(processed)

        # Extract advanced features for each text
        advanced_features_list = []
        for original_text in texts:
            try:
                adv_feats = extract_advanced_features(original_text)
                adv_array = features_to_array(adv_feats).reshape(1, -1)
                advanced_features_list.append(adv_array)
            except:
                # Fallback to zeros if extraction fails
                advanced_features_list.append(np.zeros((1, 8)))

        # Stack advanced features
        advanced_features = np.vstack(advanced_features_list)

        # Combine TF-IDF with advanced features
        combined_vectors = hstack([tfidf_vectors, advanced_features])

        # Get probabilities
        return model.predict_proba(combined_vectors)

    # Create LIME explainer
    explainer = LimeTextExplainer(class_names=class_names)

    # Generate explanation
    explanation = explainer.explain_instance(
        text,
        predict_proba_fn,
        num_features=10,  # top 10 words
        num_samples=1000  # perturbation samples
    )

    return explanation


def visualize_lime_explanation(explanation, predicted_class):
    """
    Visualize LIME explanation in Streamlit

    Args:
        explanation: LIME explanation object
        predicted_class: 0 or 1 (Fake or True)
    """
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;'>
        <h3 style='margin: 0; color: white;'>Word-Level Explanation (LIME)</h3>
        <p style='margin: 10px 0 0 0; opacity: 0.9;'>
            This shows which words most influenced the prediction.<br>
            <strong style='color: #a8ff78;'>Green</strong> = contributed to TRUE NEWS |
            <strong style='color: #ff6b6b;'>Red</strong> = contributed to FAKE NEWS
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Get explanation for predicted class
    exp_list = explanation.as_list()

    # Create DataFrame
    df = pd.DataFrame(exp_list, columns=['Word', 'Weight'])
    df['Impact'] = df['Weight'].apply(lambda x: 'Supports TRUE' if x > 0 else 'Supports FAKE')
    df['Absolute Weight'] = df['Weight'].abs()
    df = df.sort_values('Absolute Weight', ascending=False)

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        y=df['Word'],
        x=df['Weight'],
        orientation='h',
        marker=dict(
            color=df['Weight'],
            colorscale=[[0, '#ff6b6b'], [0.5, '#f0f0f0'], [1, '#a8ff78']],  # Red-Gray-Green
            cmid=0,
            showscale=True,
            colorbar=dict(
                title="Impact",
                tickvals=[-max(df['Absolute Weight']), 0, max(df['Absolute Weight'])],
                ticktext=["← FAKE", "Neutral", "TRUE →"]
            )
        ),
        text=[f"{w:.3f}" for w in df['Weight']],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.3f}<br><extra></extra>'
    ))

    fig.update_layout(
        title="Word-Level Contribution to Prediction",
        xaxis_title="Impact Score (Negative = Fake, Positive = True)",
        yaxis_title="",
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis={'categoryorder': 'total ascending'},
        font=dict(size=12),
        margin=dict(l=0, r=0, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show interpretation table
    st.markdown("#### Top Contributing Words")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🟢 Words Supporting TRUE NEWS**")
        true_words = df[df['Weight'] > 0].head(5)
        if not true_words.empty:
            for idx, row in true_words.iterrows():
                st.markdown(f"- **{row['Word']}** (+{row['Weight']:.3f})")
        else:
            st.info("No words strongly support true news")

    with col2:
        st.markdown("**🔴 Words Supporting FAKE NEWS**")
        fake_words = df[df['Weight'] < 0].head(5)
        if not fake_words.empty:
            for idx, row in fake_words.iterrows():
                st.markdown(f"- **{row['Word']}** ({row['Weight']:.3f})")
        else:
            st.info("No words strongly support fake news")

    # Show highlighted HTML
    st.markdown("---")
    st.markdown("#### 📝 Article with Highlighted Influential Words")

    try:
        html = explanation.as_html()
        import streamlit.components.v1 as components
        components.html(html, height=400, scrolling=True)
    except Exception as e:
        st.info("HTML visualization not available in this environment")


def plot_feature_importance(model, feature_names, top_n=20):
    """
    Plot top N most important features for tree-based models

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names (from vectorizer)
        top_n: Number of top features to display
    """
    if not hasattr(model, 'feature_importances_'):
        st.warning("This model doesn't support feature importance analysis")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    # Extended feature names to include advanced features
    advanced_feature_names = [
        'sentiment_polarity',
        'sentiment_subjectivity',
        'flesch_reading_ease',
        'avg_sentence_length',
        'word_count',
        'unique_word_ratio',
        'punctuation_ratio',
        'capital_ratio'
    ]

    # Combine TF-IDF feature names with advanced feature names
    all_feature_names = list(feature_names) + advanced_feature_names

    # Get feature names for top indices
    top_features = []
    for i in indices:
        if i < len(all_feature_names):
            top_features.append(all_feature_names[i])
        else:
            top_features.append(f'Feature_{i}')

    top_importances = [importances[i] for i in indices]

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=top_importances,
        y=top_features,
        orientation='h',
        marker=dict(
            color=top_importances,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=[f"{imp:.4f}" for imp in top_importances],
        textposition='auto',
        hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Top {top_n} Most Important Features (Words)",
        xaxis_title="Importance Score",
        yaxis_title="",
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show interpretation
    st.markdown("---")
    st.markdown("#### 💡 What This Means")
    st.info(f"""
    These are the **{top_n} most important words** that the Random Forest model uses to classify articles.

    - Higher importance = word has more influence on predictions
    - These words appear frequently and help distinguish fake from true news
    - The model looks for these patterns when analyzing new articles
    """)
