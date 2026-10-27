"""Streamlit frontend. All data comes from the Review Analyzer API.

Run with: streamlit run src/review_analyzer/ui/app.py
"""

import os
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.environ.get("REVIEW_ANALYZER_API_URL", "http://localhost:8000")
REVIEW_COUNTS = {"50 most recent": 50, "100 most recent": 100, "500 most recent": 500}
SENTIMENT_COLORS = {"Positive": "#4B8BBE", "Neutral": "#F28C38", "Negative": "#D9534F"}


class ApiError(Exception):
    pass


@st.cache_resource
def get_client() -> httpx.Client:
    # Long read timeout: the first analysis may download the summarization model.
    return httpx.Client(base_url=API_URL, timeout=httpx.Timeout(10.0, read=600.0))


def call_api(method: str, path: str, **params: Any) -> Any:
    try:
        response = get_client().request(method, path, params=params)
    except httpx.HTTPError as e:
        raise ApiError(f"Can't reach the API at {API_URL} ({e}). Is it running?") from e
    if response.is_error:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"{response.status_code}: {detail}")
    return response.json() if "json" in response.headers.get("content-type", "") else response.text


@st.cache_data(ttl=3600, show_spinner=False)
def search_apps(query: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = call_api("GET", "/apps/search", q=query)
    return result


def run_analysis(app_id: str, count: int) -> dict[str, Any]:
    sync = call_api("POST", f"/apps/{app_id}/sync", count=count)
    return {
        "app_id": app_id,
        "sync": sync,
        "analysis": call_api("GET", f"/apps/{app_id}/analysis", limit=count),
        "reviews": call_api("GET", f"/apps/{app_id}/reviews", limit=count)["items"],
        "csv": call_api("GET", f"/apps/{app_id}/export.csv", limit=count),
    }


def render_sentiment_chart(sentiment: dict[str, int]) -> None:
    counts = pd.DataFrame(
        [{"Sentiment": name.title(), "Count": n} for name, n in sentiment.items() if n]
    )
    fig = px.pie(
        counts,
        names="Sentiment",
        values="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(
        textinfo="percent+label",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    )
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(t=20, b=50, l=20, r=20),
    )
    st.plotly_chart(fig, width="stretch")


def render_result(result: dict[str, Any]) -> None:
    sync, analysis = result["sync"], result["analysis"]
    st.success(
        f"Analyzed {analysis['review_count']} reviews "
        f"({sync['new']} new, {sync['total_stored']} stored for this app)"
    )

    st.subheader("Sentiment (from star rating)")
    render_sentiment_chart(analysis["sentiment"])

    st.subheader("Issues in negative reviews")
    if analysis["issues"]:
        st.dataframe(
            pd.DataFrame(analysis["issues"]).rename(
                columns={"name": "Issue", "mentions": "Mentions", "suggestion": "Suggested fix"}
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Example complaints**")
        for sentence in analysis["examples"]:
            st.markdown(f"- {sentence}")
    elif analysis["summary"]:
        st.markdown(
            f"No known issue category matched. Summary of negative reviews: {analysis['summary']}"
        )
    else:
        st.info("No recurring issues found.")

    with st.expander("All reviews"):
        st.dataframe(pd.DataFrame(result["reviews"]), hide_index=True)

    st.download_button("Download CSV", result["csv"], f"{result['app_id']}_reviews.csv", "text/csv")


st.set_page_config(page_title="App Review Analyzer", layout="wide")
st.title("App Review Analyzer")
st.caption(
    "Recent Google Play reviews: sentiment breakdown, recurring complaints, and what to fix."
)

query = st.text_input("Search app by name", value="Spotify").strip()
count = REVIEW_COUNTS[st.selectbox("Reviews to analyze", list(REVIEW_COUNTS))]

if query:
    try:
        apps = search_apps(query)
    except ApiError as e:
        st.error(str(e))
        st.stop()
    if not apps:
        st.warning(f'No apps found for "{query}".')
        st.stop()

    app = st.selectbox("Select app", apps, format_func=lambda a: f"{a['title']} ({a['app_id']})")

    if st.button("Analyze", type="primary"):
        with st.spinner(f"Fetching and analyzing reviews for {app['title']}..."):
            try:
                st.session_state.result = run_analysis(app["app_id"], count)
            except ApiError as e:
                st.error(str(e))
                st.stop()

    # Kept in session state so the results survive reruns, e.g. clicking the download button.
    result = st.session_state.get("result")
    if result and result["app_id"] == app["app_id"]:
        render_result(result)
