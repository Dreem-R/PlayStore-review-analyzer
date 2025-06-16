import streamlit as st
from google_play_scraper import reviews, Sort, search
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from collections import Counter
import pandas as pd
import json
import nltk
from nltk.tokenize import sent_tokenize
import re
import plotly.express as px
import os
import glob

# Set NLTK data path to the nltk_data directory in the project
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Flag to track if NLTK tokenization is available
nltk_tokenization_available = True

# Ensure punkt_tab is available
try:
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError) as e:
    st.warning(f"Failed to load NLTK punkt_tab: {str(e)}. Sentence tokenization may not work; summarization will be used as a fallback.")
    nltk_tokenization_available = False

# Set the cache directory for Hugging Face models
cache_dir = os.path.join(os.path.dirname(__file__), 'hf_models')

# Function to find the snapshot directory containing the model files
def find_model_snapshot_dir(base_model_path):
    snapshot_dir = os.path.join(base_model_path, 'snapshots')
    if not os.path.exists(snapshot_dir):
        return None
    # Look for the first subdirectory in snapshots that contains config.json
    for subdir in glob.glob(os.path.join(snapshot_dir, '*')):
        if os.path.exists(os.path.join(subdir, 'config.json')):
            return subdir
    return None

# Load the summarization model from the local cache
try:
    # Define base paths for the models
    distilbart_path = os.path.join(cache_dir, 'sshleifer', 'distilbart-cnn-6-6')
    t5_small_path = os.path.join(cache_dir, 't5-small')

    # Find the snapshot directories
    distilbart_snapshot = find_model_snapshot_dir(distilbart_path)
    t5_small_snapshot = find_model_snapshot_dir(t5_small_path)

    # Load the primary model (distilbart)
    if distilbart_snapshot:
        model = AutoModelForSeq2SeqLM.from_pretrained(distilbart_snapshot)
        tokenizer = AutoTokenizer.from_pretrained(distilbart_snapshot)
        summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
    else:
        raise FileNotFoundError("distilbart-cnn-6-6 snapshot directory not found")

except Exception as e:
    st.error(f"Failed to load sshleifer/distilbart-cnn-6-6 model: {str(e)}")
    # Fallback to t5-small
    try:
        if t5_small_snapshot:
            model = AutoModelForSeq2SeqLM.from_pretrained(t5_small_snapshot)
            tokenizer = AutoTokenizer.from_pretrained(t5_small_snapshot)
            summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=-1)
        else:
            raise FileNotFoundError("t5-small snapshot directory not found")
    except Exception as e:
        st.error(f"Failed to load t5-small model: {str(e)}")
        st.stop()  # Stop the app if both models fail to load

def extract_key_issues(reviews):
    if not reviews:
        return ["No negative reviews to summarize."]
    
    combined_reviews = " ".join(reviews)
    
    # Try sentence tokenization if NLTK data is available
    if nltk_tokenization_available:
        try:
            sentences = sent_tokenize(combined_reviews)
        except Exception as e:
            st.warning(f"Sentence tokenization failed: {str(e)}. Falling back to summarization.")
            sentences = []
    else:
        sentences = []

    # Keyword-based issue extraction
    issue_keywords = {
        "interface": ["interface", "design", "navigation", "ui"],
        "performance": ["slow", "lag", "crash", "bug", "performance"],
        "security": ["security", "privacy", "hack", "leak"],
        "size": ["size", "storage", "space"],
        "ads": ["ads", "advertisement", "pop-up"],
        "login": ["login", "signup", "account", "verification"]
    }
    
    key_issues = []
    if sentences:
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for category, keywords in issue_keywords.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    cleaned_sentence = re.sub(r'\s+', ' ', sentence.strip())
                    if len(cleaned_sentence) > 10:
                        key_issues.append(cleaned_sentence)
                    break
    
    # Fallback to summarization if no issues are found or if tokenization failed
    if not key_issues:
        try:
            summary_result = summarizer(
                combined_reviews[:1000],
                max_length=100,
                min_length=30,
                do_sample=False
            )
            key_issues = [summary_result[0]['summary_text']]
        except Exception as e:
            key_issues = [f"Error summarizing reviews: {str(e)}"]
    
    return key_issues[:5]

def analyze_reviews(package_name, review_count):
    try:
        result, _ = reviews(
            package_name,
            lang='en',
            country='us',
            sort=Sort.NEWEST,
            count=review_count
        )
    except Exception as e:
        return None, str(e)

    data = []
    negative_reviews = []
    positive_reviews = []

    for review in result:
        content = review['content']
        rating = review['score']
        sentiment_by_rating = (
            "Positive" if rating >= 4 else
            "Neutral" if rating == 3 else
            "Negative"
        )

        if rating <= 2:
            negative_reviews.append(content)
        if rating >= 4:
            positive_reviews.append(content)

        data.append({
            "User": review['userName'],
            "Rating": rating,
            "Date": review['at'],
            "Review": content,
            "Sentiment (Rating-Based)": sentiment_by_rating
        })

    key_issues = extract_key_issues(negative_reviews)
    combined_issues = " ".join(key_issues)
    suggestions = generate_suggestions(combined_issues)

    return pd.DataFrame(data), key_issues, suggestions

def generate_suggestions(summary):
    suggestions = []
    summary_lower = summary.lower()
    
    if any(keyword in summary_lower for keyword in ["interface", "design", "navigation", "ui"]):
        suggestions.append("Improve interface consistency and streamline navigation.")
    if any(keyword in summary_lower for keyword in ["slow", "lag", "crash", "bug", "performance"]):
        suggestions.append("Optimize loading times and fix reported bugs/crashes.")
    if any(keyword in summary_lower for keyword in ["security", "privacy", "hack", "leak"]):
        suggestions.append("Enhance user data protection and address privacy concerns.")
    if any(keyword in summary_lower for keyword in ["size", "storage", "space"]):
        suggestions.append("Reduce app bloat by optimizing assets and libraries.")
    if any(keyword in summary_lower for keyword in ["ads", "advertisement", "pop-up"]):
        suggestions.append("Minimize intrusive ads and allow ad-free experience if possible.")
    if any(keyword in summary_lower for keyword in ["login", "signup", "account", "verification"]):
        suggestions.append("Simplify authentication and ensure smooth signup/login flows.")
    
    return suggestions if suggestions else ["No specific improvements identified from the summary."]

st.set_page_config(page_title="App Review Analyzer Pro", layout="wide")
st.title("📱 App Review Analyzer with AI Insights")
st.caption("Get AI-powered insights from app reviews and improve based on real user concerns.")

app_query = st.text_input("🔍 Search App by Name:", value="Facebook")
review_options = {
    "Top 50": 50,
    "Top 100": 100,
    "Top 500": 500
}
review_count_label = st.selectbox("How many reviews to analyze?", list(review_options.keys()))
review_count = review_options[review_count_label]

if app_query:
    results = search(app_query, lang="en", country="us")
    app_options = [f"{app['title']} ({app['appId']})" for app in results[:5]]
    selected_app = st.selectbox("Select App from Results:", app_options)

    if st.button("🚀 Analyze"):
        app_id = selected_app.split("(")[-1].replace(")", "")
        with st.spinner("Analyzing reviews..."):
            df, key_issues, suggestions = analyze_reviews(app_id, review_count)

        if df is None:
            st.error("Error fetching reviews.")
        else:
            st.success(f"✅ Analyzed {len(df)} reviews")

            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment (Rating-Based)"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]
            
            if not sentiment_counts.empty:
                fig_pie = px.pie(
                    sentiment_counts,
                    names="Sentiment",
                    values="Count",
                    title="Sentiment Distribution of Reviews",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#4B8BBE",
                        "Neutral": "#F28C38",
                        "Negative": "#D9534F"
                    }
                )
                fig_pie.update_traces(
                    textinfo="percent+label",
                    textposition="inside",
                    marker=dict(
                        line=dict(color="#FFFFFF", width=2)
                    ),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
                    opacity=0.9
                )
                fig_pie.update_layout(
                    title_font_size=20,
                    title_font_color="#FFFFFF",
                    font=dict(color="#FFFFFF"),
                    legend_title_text="Sentiment",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        font=dict(size=12, color="#FFFFFF"),
                        bgcolor="rgba(255,255,255,0.8)"
                    ),
                    margin=dict(t=50, b=50, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("No sentiment data available to display the chart.")

            st.subheader("🚨 Key Issues & Suggestions")
            st.markdown("**Key Issues from Negative Reviews:**")
            for issue in key_issues:
                st.markdown(f"- {issue}")
            
            if suggestions:
                st.markdown("**Suggestions for Improvement:**")
                for suggestion in suggestions:
                    st.markdown(f"- 💡 {suggestion}")
            else:
                st.info("No specific suggestions generated.")

            with st.expander("📃 Full Review Data"):
                st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download CSV", csv, "review_analysis.csv", "text/csv")