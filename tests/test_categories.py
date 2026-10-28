import pytest

from review_analyzer.categories import categorize


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("The UI is cluttered and confusing", "Interface"),
        ("It keeps crashing on startup", "Performance"),
        ("Way too many ads in the feed", "Ads"),
        ("A full-screen ad after every tap", "Ads"),
        ("I can't log in anymore", "Login"),
        ("Takes up so much storage on my phone", "Size"),
        ("Charged twice for the subscription", "Pricing"),
        # Substrings of keywords must not match.
        ("I can't build a playlist anymore", None),
        ("It loads my downloads correctly", None),
        ("Please add a widget", None),
    ],
)
def test_categorize_matches_whole_words(sentence: str, expected: str | None) -> None:
    assert categorize(sentence) == expected
