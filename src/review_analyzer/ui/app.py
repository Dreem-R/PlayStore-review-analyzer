"""Streamlit frontend.

Runs the analysis in-process by default, or talks to a running API when
REVIEW_ANALYZER_API_URL is set (as in docker-compose).

Run with: streamlit run src/review_analyzer/ui/app.py
"""

import re
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from review_analyzer.config import Settings
from review_analyzer.ui.backend import Backend, BackendError, create_backend

REVIEW_COUNTS = (100, 200, 500, 1000)
DEFAULT_COUNT = 200
SENTIMENT_COLORS = {"Positive": "#4B8BBE", "Neutral": "#F28C38", "Negative": "#D9534F"}
PACKAGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$")
PLAY_STORE_LINK = re.compile(r"[?&]id=([A-Za-z][\w.]+)")
# Characters Streamlit's markdown would otherwise treat as formatting, LaTeX ($) or emoji (:).
MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|$~<>:])")


@st.cache_resource(show_spinner="Starting up...")
def get_backend() -> Backend:
    return create_backend()


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


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
        return [get_backend().lookup(app_id)]
    return get_backend().search(query)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def app_titles(app_ids: tuple[str, ...]) -> dict[str, str]:
    titles = {}
    for app_id in app_ids:
        try:
            titles[app_id] = get_backend().lookup(app_id)["title"]
        except BackendError:
            titles[app_id] = app_id
    return titles


def run_analysis(app_id: str, count: int) -> dict[str, Any]:
    backend = get_backend()
    return {
        "app_id": app_id,
        "sync": backend.sync(app_id, count),
        "analysis": backend.analysis(app_id, count),
        "reviews": backend.reviews(app_id, count),
        "csv": backend.export_csv(app_id, count),
    }


def time_ago(timestamp: str | None) -> str:
    if not timestamp:
        return "earlier"
    then = datetime.fromisoformat(timestamp).replace(tzinfo=UTC)
    minutes = int((datetime.now(UTC) - then).total_seconds() // 60)
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes} min ago"
    return f"{minutes // 60} h ago"


def render_freshness(sync: dict[str, Any]) -> None:
    when = time_ago(sync["last_synced_at"])
    if sync["status"] == "fetched":
        st.caption(f"Fetched {sync['fetched']} reviews from the Play Store {when}.")
    elif sync["status"] == "fresh":
        st.caption(f"Using reviews fetched {when}. Apps are re-fetched every few hours.")
    else:
        st.warning(
            f"Couldn't fetch new reviews right now ({sync['message']}). "
            f"Showing the {sync['total_stored']} reviews fetched {when}."
        )


def render_summary(analysis: dict[str, Any]) -> None:
    total = analysis["review_count"]
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


def escape_markdown(text: str) -> str:
    return MARKDOWN_SPECIAL.sub(r"\\\1", text)


def highlight(content: str, sentences: list[str]) -> str:
    """The review as markdown, with the sentences that raised the issue highlighted.

    Sentences were whitespace-normalized when the review was split, so the review is too.
    """
    text = " ".join(content.split())
    spans = []
    for sentence in sentences:
        start = text.find(sentence)
        if start >= 0:
            spans.append((start, start + len(sentence)))
    parts, position = [], 0
    for start, end in sorted(spans):
        if start < position:  # overlaps a span already highlighted
            continue
        parts.append(escape_markdown(text[position:start]))
        parts.append(f":orange-background[{escape_markdown(text[start:end])}]")
        position = end
    parts.append(escape_markdown(text[position:]))
    return "".join(parts)


def review_header(review: dict[str, Any]) -> str:
    details = ["★" * review["rating"] + "☆" * (5 - review["rating"])]
    if review["app_version"]:
        details.append(f"v{review['app_version']}")
    details.append(review["created_at"][:10])
    if review["thumbs_up"]:
        details.append(f"{review['thumbs_up']} found this helpful")
    return " · ".join(details)


@st.dialog("Reviews raising this issue", width="large")
def show_issue_reviews(issue: dict[str, Any], reviews_by_id: dict[str, dict[str, Any]]) -> None:
    st.markdown(f"**{escape_markdown(issue['label'])}**")
    st.caption(
        f"{issue['reviews']} reviews, most helpful first. "
        "Highlighted: the sentences grouped into this issue."
    )
    for member in issue["members"]:
        review = reviews_by_id.get(member["review_id"])
        if review is None:
            continue
        st.caption(review_header(review))
        st.markdown(highlight(review["content"], member["sentences"]))
        st.divider()


def render_issues(analysis: dict[str, Any], reviews: list[dict[str, Any]]) -> None:
    if not analysis["issues"]:
        st.info("No recurring complaints found.")
        return
    if analysis["issue_method"] == "keywords":
        st.caption("Grouped by keyword category (the ML extra isn't installed).")
    reviews_by_id = {review["review_id"]: review for review in reviews}
    for index, issue in enumerate(analysis["issues"]):
        label = escape_markdown(issue["label"])
        title = f"**{label}** · {issue['reviews']} reviews ({issue['share']:.0%})"
        with st.expander(title):
            if issue["suggestion"]:
                st.markdown(f"*{issue['category']}:* {issue['suggestion']}")
            for example in issue["examples"]:
                st.markdown(f"> {escape_markdown(example)}")
            if st.button(f"Read all {issue['reviews']} reviews", key=f"issue-{index}"):
                show_issue_reviews(issue, reviews_by_id)


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
        st.markdown(f"> {'★' * review['rating']} {escape_markdown(review['content'])}")


def render_result(result: dict[str, Any]) -> None:
    analysis = result["analysis"]
    render_freshness(result["sync"])
    render_summary(analysis)

    st.subheader("Sentiment")
    render_sentiment(analysis)

    st.subheader("Recurring complaints")
    render_issues(analysis, result["reviews"])

    st.subheader("By app version")
    render_versions(analysis)

    render_mismatches(analysis)

    with st.expander("All reviews"):
        st.dataframe(pd.DataFrame(result["reviews"]), hide_index=True)
    st.download_button("Download CSV", result["csv"], f"{result['app_id']}_reviews.csv", "text/csv")


def show_unavailable(error: BackendError, popular: dict[str, str]) -> None:
    st.error(f"The Play Store isn't answering this demo right now ({error}).")
    if popular:
        st.info("These apps are already stored and still work: " + ", ".join(popular.values()))


def main() -> None:
    st.set_page_config(page_title="App Review Analyzer", layout="wide")
    st.title("App Review Analyzer")
    st.caption(
        "What users complain about in recent Google Play reviews, and how it changes by release."
    )

    settings = get_settings()
    popular = app_titles(tuple(settings.preload_apps)) if settings.preload_apps else {}

    def use_popular_app() -> None:
        # Picking a popular app fills the search box with its package name.
        if choice := st.session_state.get("popular"):
            st.session_state.query = choice

    st.session_state.setdefault("query", "" if popular else "Spotify")
    st.text_input("App name, package name or Play Store link", key="query")
    if popular:
        st.pills(
            "Or pick one that's already analyzed (instant)",
            list(popular),
            format_func=lambda app_id: popular[app_id],
            key="popular",
            on_change=use_popular_app,
        )
    counts = [c for c in REVIEW_COUNTS if c <= settings.max_sync_count]
    count = st.selectbox(
        "Reviews to analyze",
        counts,
        index=counts.index(DEFAULT_COUNT) if DEFAULT_COUNT in counts else 0,
        format_func=lambda c: f"{c} most recent",
    )

    query = st.session_state.query.strip()
    if not query:
        st.stop()
    try:
        apps = find_apps(query)
    except BackendError as e:
        if e.kind == "unavailable":
            show_unavailable(e, popular)
        else:
            st.error(str(e))
        st.stop()
    if not apps:
        st.warning(f'No apps found for "{query}".')
        st.stop()

    app = st.selectbox("Select app", apps, format_func=lambda a: f"{a['title']} ({a['app_id']})")

    if st.button("Analyze", type="primary"):
        with st.spinner(
            f"Analyzing reviews for {app['title']}. An app's first analysis takes up to a minute "
            "while every review is scored; after that it's instant."
        ):
            try:
                st.session_state.result = run_analysis(app["app_id"], count)
            except BackendError as e:
                if e.kind == "unavailable":
                    show_unavailable(e, popular)
                else:
                    st.error(str(e))
                st.stop()

    # Kept in session state so the results survive reruns, e.g. clicking the download button.
    result = st.session_state.get("result")
    if result and result["app_id"] == app["app_id"]:
        render_result(result)


if __name__ == "__main__":
    main()
