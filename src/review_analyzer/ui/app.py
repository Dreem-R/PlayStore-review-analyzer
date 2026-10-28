"""Streamlit frontend. All data comes from the Review Analyzer API.

Run with: streamlit run src/review_analyzer/ui/app.py
"""

import os
import re
from typing import Any

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_URL = os.environ.get("REVIEW_ANALYZER_API_URL", "http://localhost:8000")
REVIEW_COUNTS = {"100 most recent": 100, "500 most recent": 500, "1000 most recent": 1000}
SENTIMENT_COLORS = {"Positive": "#4B8BBE", "Neutral": "#F28C38", "Negative": "#D9534F"}
PACKAGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
PLAY_STORE_LINK = re.compile(r"[?&]id=([A-Za-z][\w.]+)")


class ApiError(Exception):
    pass


@st.cache_resource
def get_client() -> httpx.Client:
    # Long read timeout: the first analysis downloads the models.
    return httpx.Client(base_url=API_URL, timeout=httpx.Timeout(10.0, read=900.0))


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


def parse_app_id(text: str) -> str | None:
    """Accept a package name or a Play Store link instead of a search term."""
    if PACKAGE_NAME.match(text):
        return text
    if match := PLAY_STORE_LINK.search(text):
        return match.group(1)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def find_apps(query: str) -> list[dict[str, Any]]:
    if app_id := parse_app_id(query):
        return [call_api("GET", f"/apps/{app_id}")]
    result: list[dict[str, Any]] = call_api("GET", "/apps/search", q=query)
    return result


def run_analysis(app_id: str, count: int) -> dict[str, Any]:
    sync = call_api("POST", f"/apps/{app_id}/sync", count=count)
    return {
        "app_id": app_id,
        "sync": sync,
        "analysis": call_api("GET", f"/apps/{app_id}/analysis", limit=count),
        "reviews": call_api("GET", f"/apps/{app_id}/reviews", limit=min(count, 500))["items"],
        "csv": call_api("GET", f"/apps/{app_id}/export.csv", limit=count),
    }


def render_summary(sync: dict[str, Any], analysis: dict[str, Any]) -> None:
    total = analysis["review_count"]
    st.caption(f"{sync['new']} new reviews fetched, {sync['total_stored']} stored for this app.")
    columns = st.columns(4)
    columns[0].metric("Reviews analyzed", total)
    columns[1].metric("Complaints", f"{analysis['complaint_count'] / total:.0%}")
    columns[2].metric("Distinct issues", len(analysis["issues"]))
    columns[3].metric(
        "Hidden complaints",
        len(analysis["mismatches"]),
        help="4-5 star reviews whose text is clearly negative.",
    )


def render_sentiment(analysis: dict[str, Any]) -> None:
    rows = [
        {"Source": "Star rating", "Sentiment": name.title(), "Reviews": n}
        for name, n in analysis["rating_sentiment"].items()
    ]
    if analysis["text_sentiment"]:
        rows += [
            {"Source": "Review text", "Sentiment": name.title(), "Reviews": n}
            for name, n in analysis["text_sentiment"].items()
        ]
    fig = px.bar(
        pd.DataFrame(rows),
        x="Reviews",
        y="Source",
        color="Sentiment",
        orientation="h",
        color_discrete_map=SENTIMENT_COLORS,
        category_orders={"Sentiment": list(SENTIMENT_COLORS)},
    )
    fig.update_layout(height=220, margin=dict(t=10, b=10, l=10, r=10), yaxis_title=None)
    st.plotly_chart(fig, width="stretch")


def render_issues(analysis: dict[str, Any]) -> None:
    if not analysis["issues"]:
        st.info("No recurring complaints found.")
        return
    if analysis["issue_method"] == "keywords":
        st.caption("Grouped by keyword category (the ML extra isn't installed on the API).")
    for issue in analysis["issues"]:
        title = f"**{issue['label']}** · {issue['reviews']} reviews ({issue['share']:.0%})"
        with st.expander(title):
            if issue["suggestion"]:
                st.markdown(f"*{issue['category']}:* {issue['suggestion']}")
            for example in issue["examples"]:
                st.markdown(f"> {example}")


def render_versions(analysis: dict[str, Any]) -> None:
    versions = pd.DataFrame(analysis["versions"])
    if len(versions) < 2:
        st.info("Not enough reviews on different app versions to show a trend.")
        return
    fig = px.line(
        versions,
        x="version",
        y="negative_share",
        markers=True,
        hover_data={"reviews": True, "average_rating": True},
        labels={"version": "App version", "negative_share": "Negative reviews"},
    )
    fig.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    issues = analysis["issues"]
    known = list(versions["version"])
    matrix = pd.DataFrame(
        [[issue["by_version"].get(v, 0) for v in known] for issue in issues],
        index=[issue["label"] for issue in issues],
        columns=known,
    )
    if not matrix.empty and matrix.to_numpy().sum():
        st.markdown("**Issues by version** (reviews raising each issue)")
        heatmap = px.imshow(matrix, text_auto=True, aspect="auto", color_continuous_scale="Reds")
        heatmap.update_layout(margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(heatmap, width="stretch")


def render_mismatches(analysis: dict[str, Any]) -> None:
    if not analysis["mismatches"]:
        return
    st.subheader("Complaints hidden in high ratings")
    st.caption("Rated 4-5 stars, but the text is clearly negative. Star averages miss these.")
    for review in analysis["mismatches"]:
        st.markdown(f"> {'★' * review['rating']} {review['content']}")


def render_result(result: dict[str, Any]) -> None:
    analysis = result["analysis"]
    render_summary(result["sync"], analysis)

    st.subheader("Sentiment")
    render_sentiment(analysis)

    st.subheader("Recurring complaints")
    render_issues(analysis)

    st.subheader("By app version")
    render_versions(analysis)

    render_mismatches(analysis)

    with st.expander("All reviews"):
        st.dataframe(pd.DataFrame(result["reviews"]), hide_index=True)
    st.download_button("Download CSV", result["csv"], f"{result['app_id']}_reviews.csv", "text/csv")


st.set_page_config(page_title="App Review Analyzer", layout="wide")
st.title("App Review Analyzer")
st.caption(
    "What users complain about in recent Google Play reviews, and how it changes by release."
)

query = st.text_input("App name, package name or Play Store link", value="Spotify").strip()
count = REVIEW_COUNTS[st.selectbox("Reviews to analyze", list(REVIEW_COUNTS))]

if query:
    try:
        apps = find_apps(query)
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
