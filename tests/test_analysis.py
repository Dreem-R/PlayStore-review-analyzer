from datetime import datetime

import pytest

from review_analyzer.analysis import (
    analyze,
    complaint_sentences,
    find_mismatches,
    is_complaint,
    sentiment_from_rating,
    split_sentences,
    version_stats,
)
from review_analyzer.schemas import Review, Sentiment
from tests.conftest import FakeEmbedder, FakeSentiment


def make_review(
    review_id: str,
    rating: int,
    content: str = "",
    version: str | None = "1.0.0",
    text_sentiment: Sentiment | None = None,
    score: float | None = None,
    thumbs_up: int = 0,
) -> Review:
    return Review(
        review_id=review_id,
        app_id="com.example.notes",
        user_name="user",
        rating=rating,
        content=content,
        created_at=datetime(2026, 10, 1),
        app_version=version,
        thumbs_up=thumbs_up,
        text_sentiment=text_sentiment,
        sentiment_score=score,
    )


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_sentiment_from_rating(rating: int, expected: str) -> None:
    assert sentiment_from_rating(rating) == expected


def test_split_sentences_drops_fragments_and_normalizes_whitespace() -> None:
    text = "Crashes   constantly on my phone.  Ugh!\nThe new   update broke search."
    assert split_sentences(text) == [
        "Crashes constantly on my phone.",
        "The new update broke search.",
    ]


def test_complaints_include_high_ratings_with_negative_text() -> None:
    assert is_complaint(make_review("a", 1))
    assert is_complaint(make_review("b", 5, text_sentiment="negative"))
    assert not is_complaint(make_review("c", 5, text_sentiment="positive"))
    assert not is_complaint(make_review("d", 3))


def test_keyword_analysis_without_models(reviews: list[Review]) -> None:
    result = analyze("com.example.notes", reviews)

    assert result.issue_method == "keywords"
    assert result.review_count == 10
    assert result.rating_sentiment.model_dump() == {"positive": 4, "neutral": 1, "negative": 5}
    assert result.text_sentiment is None
    assert result.complaint_count == 5
    assert {i.label: i.reviews for i in result.issues} == {"Performance": 2, "Ads": 1, "Login": 1}
    assert result.issues[0].label == "Performance"
    ads = next(i for i in result.issues if i.label == "Ads")
    assert ads.suggestion is not None
    assert len(ads.examples) == 2  # both sentences of the one ads review
    assert ads.share == pytest.approx(1 / 5)


def test_clustering_groups_complaints_by_topic() -> None:
    reviews = [
        make_review("1", 1, "Way too many ads now.", version="2.0"),
        make_review("2", 2, "The ads after every song are unbearable.", version="2.1"),
        make_review("3", 1, "Ads ads ads, every few minutes.", version="2.1"),
        make_review("4", 1, "It crashes on startup every time.", version="2.1"),
        make_review("5", 2, "Constant crashes since the update.", version="2.1"),
        make_review("6", 1, "Customer support never answered me."),
    ]

    result = analyze("com.example.notes", reviews, FakeEmbedder())

    assert result.issue_method == "clustering"
    ads, crashes = result.issues
    assert ads.reviews == 3 and ads.category == "Ads" and "ads" in ads.label
    assert crashes.reviews == 2 and crashes.category == "Performance"
    assert ads.by_version == {"2.0": 1, "2.1": 2}
    # The support complaint stands alone, so it isn't a recurring issue.
    assert all("support" not in e for issue in result.issues for e in issue.examples)


def test_positive_sentences_are_dropped_from_complaints() -> None:
    review = make_review("1", 2, "I love the playlists. But the app keeps crashing on startup.")
    sentences = complaint_sentences([review], FakeSentiment())
    assert [s.text for s in sentences] == ["But the app keeps crashing on startup."]


def test_text_sentiment_counts_when_reviews_are_scored() -> None:
    reviews = [
        make_review("1", 5, "great", text_sentiment="positive", score=0.9),
        make_review("2", 5, "crashes constantly", text_sentiment="negative", score=0.95),
        make_review("3", 1, "meh", text_sentiment="neutral", score=0.5),
    ]
    result = analyze("com.example.notes", reviews)
    assert result.text_sentiment is not None
    assert result.text_sentiment.model_dump() == {"positive": 1, "neutral": 1, "negative": 1}
    assert result.complaint_count == 2


def test_mismatches_are_confident_negative_text_with_high_ratings() -> None:
    reviews = [
        make_review("hidden", 5, text_sentiment="negative", score=0.95, thumbs_up=1),
        make_review("popular", 4, text_sentiment="negative", score=0.85, thumbs_up=40),
        make_review("unsure", 5, text_sentiment="negative", score=0.55),
        make_review("honest", 1, text_sentiment="negative", score=0.99),
        make_review("happy", 5, text_sentiment="positive", score=0.99),
    ]
    assert [r.review_id for r in find_mismatches(reviews)] == ["popular", "hidden"]


def test_version_stats_order_and_minimum_sample() -> None:
    reviews = (
        [make_review(f"a{i}", 5, version="3.9.0") for i in range(5)]
        + [make_review(f"b{i}", 1 if i < 3 else 4, version="3.10.0") for i in range(6)]
        + [make_review(f"c{i}", 1, version="3.10.1") for i in range(2)]  # too few
        + [make_review(f"d{i}", 1, version=None) for i in range(5)]
    )

    stats = version_stats(reviews)

    assert [s.version for s in stats] == ["3.9.0", "3.10.0"]  # numeric, not string order
    assert stats[1].reviews == 6
    assert stats[1].average_rating == 2.5
    assert stats[1].negative_share == 0.5


def test_version_negative_share_prefers_text_sentiment() -> None:
    reviews = [make_review(str(i), 5, text_sentiment="negative", score=0.9) for i in range(5)]
    assert version_stats(reviews)[0].negative_share == 1.0


def test_analyze_empty() -> None:
    result = analyze("com.example.notes", [])
    assert result.review_count == 0
    assert result.issues == []
    assert result.versions == []
