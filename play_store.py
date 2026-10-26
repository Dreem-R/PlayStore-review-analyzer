import re

import pandas as pd
import plotly.express as px
import streamlit as st
from google_play_scraper import Sort, reviews, search
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-6-6"
MAX_ISSUES = 5

# Each category maps to the keywords that signal it and the suggestion shown when it appears.
ISSUE_CATEGORIES = {
    "Interface": (
        ["interface", "design", "navigation", "ui", "ux"],
        "Improve interface consistency and streamline navigation.",
    ),
    "Performance": (
        ["slow", "lag", "laggy", "lagging", "crash", "freeze", "freezing", "bug", "buggy",
         "glitch", "glitchy", "performance"],
        "Optimize loading times and fix reported bugs/crashes.",
    ),
    "Security": (
        ["security", "privacy", "hack", "leak", "scam"],
        "Enhance user data protection and address privacy concerns.",
    ),
    "Size": (
        ["size", "storage", "space"],
        "Reduce app size by optimizing assets and libraries.",
    ),
    "Ads": (
        ["ad", "ads", "advert", "advertisement", "pop-up", "popup"],
        "Reduce intrusive ads or offer an ad-free option.",
    ),
    "Login": (
        ["login", "log in", "signup", "sign up", "account", "verification", "otp"],
        "Simplify authentication and fix signup/login flows.",
    ),
}

# Whole-word match with simple inflections, so "ui" doesn't match "build" and "ads" doesn't match "loads".
CATEGORY_PATTERNS = {
    category: re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in keywords) + r")(?:s|es|ed|ing)?\b",
        re.IGNORECASE,
    )
    for category, (keywords, _) in ISSUE_CATEGORIES.items()
}

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@st.cache_resource(show_spinner="Loading summarization model (first run downloads ~500 MB)...")
def load_summarizer():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    return tokenizer, model


def summarize(text, max_length=100, min_length=30):
    tokenizer, model = load_summarizer()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    output = model.generate(**inputs, max_length=max_length, min_length=min_length, num_beams=4)
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


@st.cache_data(ttl=3600, show_spinner=False)
def search_apps(query):
    # The scraper returns appId=None for the "featured" top result; it can't be fetched, so skip it.
    results = search(query, lang="en", country="us")
    return [(app["title"], app["appId"]) for app in results if app.get("appId")][:5]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reviews(package_name, count):
    result, _ = reviews(package_name, lang="en", country="us", sort=Sort.NEWEST, count=count)
    return result


def sentiment_from_rating(rating):
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def extract_key_issues(negative_reviews):
    """Return (issue sentences, matched categories ordered by frequency)."""
    if not negative_reviews:
        return ["No negative reviews to summarize."], []

    issues = []
    category_counts = {}
    for review in negative_reviews:
        for sentence in SENTENCE_SPLIT.split(review):
            sentence = " ".join(sentence.split())
            if len(sentence) <= 10:
                continue
            for category, pattern in CATEGORY_PATTERNS.items():
                if pattern.search(sentence):
                    category_counts[category] = category_counts.get(category, 0) + 1
                    issues.append(sentence)
                    break

    categories = sorted(category_counts, key=category_counts.get, reverse=True)
    if issues:
        return issues[:MAX_ISSUES], categories

    # No keyword hits: summarize the negative reviews instead.
    try:
        return [summarize(" ".join(negative_reviews))], categories
    except Exception as e:
        return [f"Could not summarize reviews: {e}"], categories


def analyze_reviews(raw_reviews):
    data = []
    negative_reviews = []
    for review in raw_reviews:
        content = review["content"] or ""
        rating = review["score"]
        if rating <= 2:
            negative_reviews.append(content)
        data.append({
            "User": review["userName"],
            "Rating": rating,
            "Date": review["at"],
            "Review": content,
            "Sentiment (Rating-Based)": sentiment_from_rating(rating),
        })

    key_issues, categories = extract_key_issues(negative_reviews)
    suggestions = [ISSUE_CATEGORIES[c][1] for c in categories]
    return pd.DataFrame(data), key_issues, suggestions


def render_sentiment_chart(df):
    counts = df["Sentiment (Rating-Based)"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map={"Positive": "#4B8BBE", "Neutral": "#F28C38", "Negative": "#D9534F"},
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=20, b=50, l=20, r=20),
    )
    st.plotly_chart(fig, width="stretch")


st.set_page_config(page_title="App Review Analyzer", layout="wide")
st.title("App Review Analyzer")
st.caption("Summarize recent Google Play reviews: sentiment breakdown, recurring complaints, and what to fix.")

app_query = st.text_input("Search app by name", value="Facebook").strip()
review_options = {"50 most recent": 50, "100 most recent": 100, "500 most recent": 500}
review_count = review_options[st.selectbox("Reviews to analyze", list(review_options))]

if app_query:
    try:
        apps = search_apps(app_query)
    except Exception as e:
        st.error(f"Play Store search failed: {e}")
        st.stop()

    if not apps:
        st.warning(f'No apps found for "{app_query}".')
        st.stop()

    title, app_id = st.selectbox(
        "Select app",
        apps,
        format_func=lambda app: f"{app[0]} ({app[1]})",
    )

    if st.button("Analyze", type="primary"):
        with st.spinner(f"Fetching and analyzing reviews for {title}..."):
            try:
                raw_reviews = fetch_reviews(app_id, review_count)
            except Exception as e:
                st.error(f"Could not fetch reviews for {app_id}: {e}")
                st.stop()
            if not raw_reviews:
                st.warning(f"{title} has no English reviews in the US store.")
                st.stop()
            df, key_issues, suggestions = analyze_reviews(raw_reviews)

        st.success(f"Analyzed {len(df)} reviews")

        st.subheader("Sentiment distribution")
        render_sentiment_chart(df)

        st.subheader("Key issues from negative reviews")
        for issue in key_issues:
            st.markdown(f"- {issue}")

        st.subheader("Suggested improvements")
        if suggestions:
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        else:
            st.info("No specific improvements identified.")

        with st.expander("Full review data"):
            st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            f"{app_id}_reviews.csv",
            "text/csv",
        )
