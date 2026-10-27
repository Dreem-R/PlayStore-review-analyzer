"""Rating-based sentiment and keyword-based issue detection over a batch of reviews."""

import logging
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from review_analyzer.schemas import Analysis, IssueCategory, Review, SentimentCounts

logger = logging.getLogger(__name__)

Sentiment = Literal["positive", "neutral", "negative"]
Summarizer = Callable[[str], str]

MAX_EXAMPLES = 5
MIN_SENTENCE_LENGTH = 10
SUMMARY_INPUT_CHARS = 4000


@dataclass(frozen=True)
class Category:
    name: str
    keywords: tuple[str, ...]
    suggestion: str


CATEGORIES = (
    Category(
        "Interface",
        ("interface", "design", "navigation", "ui", "ux"),
        "Improve interface consistency and streamline navigation.",
    ),
    Category(
        "Performance",
        (
            "slow",
            "lag",
            "laggy",
            "lagging",
            "crash",
            "freeze",
            "freezing",
            "bug",
            "buggy",
            "glitch",
            "glitchy",
            "performance",
        ),
        "Optimize loading times and fix reported bugs/crashes.",
    ),
    Category(
        "Security",
        ("security", "privacy", "hack", "leak", "scam"),
        "Enhance user data protection and address privacy concerns.",
    ),
    Category(
        "Size",
        ("size", "storage", "space"),
        "Reduce app size by optimizing assets and libraries.",
    ),
    Category(
        "Ads",
        ("ad", "ads", "advert", "advertisement", "pop-up", "popup"),
        "Reduce intrusive ads or offer an ad-free option.",
    ),
    Category(
        "Login",
        ("login", "log in", "signup", "sign up", "account", "verification", "otp"),
        "Simplify authentication and fix signup/login flows.",
    ),
)
_SUGGESTIONS = {c.name: c.suggestion for c in CATEGORIES}

# Whole-word match with simple inflections, so "ui" doesn't match "build"
# and "ads" doesn't match "loads".
_PATTERNS = {
    c.name: re.compile(
        r"\b(?:" + "|".join(re.escape(k) for k in c.keywords) + r")(?:s|es|ed|ing)?\b",
        re.IGNORECASE,
    )
    for c in CATEGORIES
}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def sentiment_from_rating(rating: int) -> Sentiment:
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"


def split_sentences(text: str) -> list[str]:
    sentences = (" ".join(s.split()) for s in _SENTENCE_SPLIT.split(text))
    return [s for s in sentences if len(s) > MIN_SENTENCE_LENGTH]


def categorize(sentence: str) -> str | None:
    """Return the first category whose keywords appear in the sentence."""
    for name, pattern in _PATTERNS.items():
        if pattern.search(sentence):
            return name
    return None


def analyze(
    app_id: str, reviews: Sequence[Review], summarizer: Summarizer | None = None
) -> Analysis:
    sentiment = Counter(sentiment_from_rating(r.rating) for r in reviews)
    negative = [r.content for r in reviews if sentiment_from_rating(r.rating) == "negative"]

    mentions: Counter[str] = Counter()
    examples: list[str] = []
    for text in negative:
        for sentence in split_sentences(text):
            category = categorize(sentence)
            if category is None:
                continue
            mentions[category] += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append(sentence)

    summary = None
    if negative and not mentions and summarizer is not None:
        try:
            summary = summarizer(" ".join(negative)[:SUMMARY_INPUT_CHARS])
        except Exception:
            logger.exception("Summarizing negative reviews for %s failed", app_id)

    return Analysis(
        app_id=app_id,
        review_count=len(reviews),
        sentiment=SentimentCounts(
            positive=sentiment["positive"],
            neutral=sentiment["neutral"],
            negative=sentiment["negative"],
        ),
        issues=[
            IssueCategory(name=name, mentions=count, suggestion=_SUGGESTIONS[name])
            for name, count in mentions.most_common()
        ],
        examples=examples,
        summary=summary,
    )
