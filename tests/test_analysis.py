import pytest

from review_analyzer.analysis import analyze, categorize, sentiment_from_rating, split_sentences
from review_analyzer.schemas import Review


@pytest.mark.parametrize(
    ("rating", "expected"),
    [(1, "negative"), (2, "negative"), (3, "neutral"), (4, "positive"), (5, "positive")],
)
def test_sentiment_from_rating(rating: int, expected: str) -> None:
    assert sentiment_from_rating(rating) == expected


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The UI is cluttered and confusing", "Interface"),
        ("It keeps crashing on startup", "Performance"),
        ("Way too many ads in the feed", "Ads"),
        ("A full-screen ad after every tap", "Ads"),
        ("I can't log in anymore", "Login"),
        ("Takes up so much storage on my phone", "Size"),
        # Substrings of keywords must not match.
        ("I can't build a playlist anymore", None),
        ("It loads my downloads correctly", None),
        ("Please add a widget", None),
    ],
)
def test_categorize_matches_whole_words(sentence: str, expected: str | None) -> None:
    assert categorize(sentence) == expected


def test_split_sentences_drops_fragments_and_normalizes_whitespace() -> None:
    text = "Crashes   constantly on my phone.  Ugh!\nThe new   update broke search."
    assert split_sentences(text) == [
        "Crashes constantly on my phone.",
        "The new update broke search.",
    ]


def test_analyze_counts_sentiment_and_issues(reviews: list[Review]) -> None:
    result = analyze("com.example.notes", reviews)

    assert result.review_count == 10
    assert result.sentiment.model_dump() == {"positive": 4, "neutral": 1, "negative": 5}
    assert {i.name: i.mentions for i in result.issues} == {"Performance": 2, "Ads": 2, "Login": 1}
    mentions = [i.mentions for i in result.issues]
    assert mentions == sorted(mentions, reverse=True)
    assert "The app crashes every time I open a note with images." in result.examples
    assert result.summary is None


def test_analyze_ignores_keywords_in_positive_reviews(reviews: list[Review]) -> None:
    positive_only = [r for r in reviews if r.rating >= 4]
    result = analyze("com.example.notes", positive_only)
    assert result.issues == []
    assert result.examples == []


def test_analyze_summarizes_when_no_category_matches(reviews: list[Review]) -> None:
    unmatched = [r for r in reviews if r.review_id == "r09"]
    calls: list[str] = []

    def summarizer(text: str) -> str:
        calls.append(text)
        return "Users lost notes after the update."

    result = analyze("com.example.notes", unmatched, summarizer)

    assert result.issues == []
    assert result.summary == "Users lost notes after the update."
    assert calls == [unmatched[0].content]


def test_analyze_survives_summarizer_failure(reviews: list[Review]) -> None:
    unmatched = [r for r in reviews if r.review_id == "r09"]

    def broken(text: str) -> str:
        raise RuntimeError("model failed to load")

    assert analyze("com.example.notes", unmatched, broken).summary is None


def test_analyze_empty() -> None:
    result = analyze("com.example.notes", [])
    assert result.review_count == 0
    assert result.issues == []
