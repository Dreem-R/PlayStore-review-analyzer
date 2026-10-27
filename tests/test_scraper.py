from datetime import datetime
from typing import Any

import pytest

from review_analyzer import scraper
from review_analyzer.scraper import GooglePlayClient, UpstreamError


def test_search_skips_results_without_app_id(monkeypatch: pytest.MonkeyPatch) -> None:
    results = [
        {"appId": None, "title": "Featured App"},
        {"appId": "com.example.one", "title": "One", "developer": "Dev", "score": 4.5},
        {"appId": "com.example.two", "title": "Two"},
    ]
    monkeypatch.setattr(scraper, "search", lambda *args, **kwargs: results)

    apps = GooglePlayClient().search("example", limit=5)

    assert [a.app_id for a in apps] == ["com.example.one", "com.example.two"]
    assert apps[0].score == 4.5


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


def test_scraper_errors_become_upstream_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise ConnectionError("network down")

    monkeypatch.setattr(scraper, "reviews", boom)

    with pytest.raises(UpstreamError, match="network down"):
        GooglePlayClient().fetch_reviews("com.example.one", count=10)
