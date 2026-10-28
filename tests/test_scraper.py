from datetime import datetime
from typing import Any

import pytest
from google_play_scraper.exceptions import NotFoundError

from review_analyzer import scraper
from review_analyzer.scraper import AppNotFoundError, GooglePlayClient, UpstreamError

SEARCH_RESULTS = [
    {"appId": None, "title": "Example: Music"},
    {"appId": "com.example.lite", "title": "Example Lite", "developer": "Dev", "score": 4.5},
    {"appId": "com.example.kids", "title": "Example Kids"},
]
DETAILS = {
    "com.example.music": {"appId": "com.example.music", "title": "Example: Music"},
    "com.other.ad": {"appId": "com.other.ad", "title": "Sponsored Thing"},
}


def fake_details(app_id: str, **kwargs: Any) -> dict[str, Any]:
    if app_id not in DETAILS:
        raise NotFoundError("App not found(404).")
    return DETAILS[app_id]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> GooglePlayClient:
    monkeypatch.setattr(scraper, "search", lambda *args, **kwargs: SEARCH_RESULTS)
    monkeypatch.setattr(scraper, "app", fake_details)
    return GooglePlayClient()


def test_search_resolves_featured_result_from_search_page(
    client: GooglePlayClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The page links an unrelated app first; its title doesn't match, so it's skipped.
    page_ids = ["com.other.ad", "com.example.music", "com.example.lite"]
    monkeypatch.setattr(GooglePlayClient, "_search_page_app_ids", lambda self, q: page_ids)

    apps = client.search("example", limit=5)

    assert [a.app_id for a in apps] == ["com.example.music", "com.example.lite", "com.example.kids"]


def test_search_skips_featured_result_it_cannot_resolve(
    client: GooglePlayClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def offline(self: GooglePlayClient, query: str) -> list[str]:
        raise OSError("network down")

    monkeypatch.setattr(GooglePlayClient, "_search_page_app_ids", offline)

    assert [a.app_id for a in client.search("example", limit=5)] == [
        "com.example.lite",
        "com.example.kids",
    ]


def test_search_respects_limit(client: GooglePlayClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(GooglePlayClient, "_search_page_app_ids", lambda self, q: [])
    assert len(client.search("example", limit=1)) == 1


def test_lookup(client: GooglePlayClient) -> None:
    assert client.lookup("com.example.music").title == "Example: Music"


def test_lookup_unknown_app(client: GooglePlayClient) -> None:
    with pytest.raises(AppNotFoundError):
        client.lookup("com.nope.nope")


def test_fetch_reviews_maps_scraper_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    raw: dict[str, Any] = {
        "reviewId": "abc",
        "userName": "Sam",
        "content": None,
        "score": 2,
        "thumbsUpCount": 3,
        "reviewCreatedVersion": None,
        "appVersion": "1.4.0",
        "at": datetime(2026, 10, 1, 12, 0),
    }
    monkeypatch.setattr(scraper, "reviews", lambda *args, **kwargs: ([raw], None))

    [review] = GooglePlayClient().fetch_reviews("com.example.one", count=1)

    assert review.review_id == "abc"
    assert review.app_id == "com.example.one"
    assert review.content == ""
    assert review.app_version == "1.4.0"
    assert review.thumbs_up == 3
    assert review.text_sentiment is None


def test_scraper_errors_become_upstream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("network down")

    monkeypatch.setattr(scraper, "reviews", boom)

    with pytest.raises(UpstreamError, match="network down"):
        GooglePlayClient().fetch_reviews("com.example.one", count=10)
