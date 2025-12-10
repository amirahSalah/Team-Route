
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
# Import functions from app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import preprocess_row
import app as prep_app


st.set_page_config(page_title="Real-Time Sentiment Dashboard", layout="wide")
st.title("Real-Time Sentiment Analysis Dashboard")

# Load models and data
models_dir = Path('models')
try:
    tfidf_vec = joblib.load(models_dir / 'rf_tfidf_ngram12_vectorizer.joblib') 
    rf_clf = joblib.load(models_dir / 'rf_clf.joblib')
    proc_df = pd.read_parquet(models_dir / 'processed_df.parquet')
    kmeans = joblib.load(models_dir / 'kmeans_topics.joblib') if (models_dir / 'kmeans_topics.joblib').exists() else None
    topic_vec = joblib.load(models_dir / 'tfidf_for_topics.joblib') if (models_dir / 'tfidf_for_topics.joblib').exists() else None
    X_vec_for_topics = joblib.load(models_dir / 'X_vec_for_topics.joblib') if (models_dir / 'X_vec_for_topics.joblib').exists() else None
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    tfidf_vec = None
    rf_clf = None
    proc_df = None
    kmeans = None
    topic_vec = None
    X_vec_for_topics = None

tab0,tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Preprocessing",
    "Sentiment Analysis", 
    "Topic Assignment", 
    "Trend Visualization", 
    "Alerts", 
    "Export"
])
with tab0:
    prep_app.display_preproccess_content()
with tab1:
    st.header("Instant Sentiment Analysis")
    user_text = st.text_area("Enter tweet or text for analysis:")
    # Check each component separately
    if not user_text:
        st.warning("Please enter some text")
    elif tfidf_vec is None:
        st.error("TF-IDF vectorizer not loaded")
    elif rf_clf is None:
        st.error("Random Forest model not loaded")
    else:
        # All good, proceed
        res = preprocess_row(user_text)
        vec = tfidf_vec.transform([res['cleaned']])
        pred = rf_clf.predict(vec)[0]
        st.write(f"Predicted Sentiment: **{pred}**")

with tab2:
    st.header("Topic Assignment & Similar Content")
    if kmeans and topic_vec and X_vec_for_topics is not None and proc_df is not None:
        user_text_topic = st.text_area("Enter text for topic assignment:", key="topic_input")
        if user_text_topic:
            topic_vec_input = topic_vec.transform([user_text_topic])
            topic_pred = kmeans.predict(topic_vec_input)[0]
            st.write(f"Assigned Topic: **{topic_pred}**")
            from sklearn.metrics.pairwise import cosine_similarity
            sim_scores = cosine_similarity(topic_vec_input, X_vec_for_topics)[0]
            top_idx = np.argsort(sim_scores)[-5:][::-1]
            st.write("Most similar tweets:")
            for idx in top_idx:
                st.write(proc_df.iloc[idx]['Tweet'])
    else:
        st.info("Topic assignment model/data not available.")

with tab3:
    st.header("Sentiment Trend Visualization")
    if proc_df is not None:
        proc_df['time'] = np.arange(len(proc_df))
        fig = px.line(proc_df.groupby('time').Sentiment.value_counts().unstack().fillna(0), title="Sentiment Trend Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for trend visualization.")

with tab4:
    st.header("Alerts")
    uploaded = st.file_uploader("Upload processed CSV with named columns including [\"Sentiment\"] column to analyze Negative sentiment spikes", type=["csv"]) 

    if uploaded is not None and pd is not None:
        df = pd.read_csv(uploaded)
        st.write(df.head(20))
        st.write("Sentiment counts:")
        if 'Sentiment' in df.columns:
            st.bar_chart(df['Sentiment'].value_counts())
        else:
            st.info("No `sentiment` column found in uploaded file.")
        neg_counts = df['Sentiment'].value_counts().get('Negative', 0)
        st.write(f"{df['Sentiment'].value_counts().get('Negative', 0)} negative samples out of {len(df)} total samples.")
        total = len(df)
        neg_pct = neg_counts / total if total > 0 else 0
        if neg_pct > 0.2: # define sensitive threshold for negative sentiment spike
            st.error(f"Alert: Negative sentiment spike detected! ({neg_pct:.1%})")
        else:
            st.success("No negative sentiment spike detected.")
    else:
        if uploaded is not None:
            st.warning("Pandas not available to read the CSV. Install pandas in your environment.")

    st.write(f"=============================================")
    st.write(f"=============================================")
    st.write(f"=============================================")
    st.header("Our twitter processed samples:")
    st.write(proc_df.head(20))
    if proc_df is not None:
        neg_counts = proc_df['Sentiment'].value_counts().get('Negative', 0)
        st.write(f"{proc_df['Sentiment'].value_counts().get('Negative', 0)} negative samples out of {len(proc_df)} total samples.")
        total = len(proc_df)
        neg_pct = neg_counts / total if total > 0 else 0
        if neg_pct > 0.4:
            st.error(f"Alert: Negative sentiment spike detected! ({neg_pct:.1%})")
        else:
            st.success("No negative sentiment spike detected.")
    else:
        st.info("No data available for alerts.")

with tab5:
    st.header("Export Social Media Report")
    if proc_df is not None:
        st.download_button("Download full dataset as CSV", proc_df.to_csv(index=False), "sentiment_report.csv")
    else:
        st.info("No data available for export.")
