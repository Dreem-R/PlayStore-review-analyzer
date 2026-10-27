"""Thin wrapper around google-play-scraper that returns typed models."""

from typing import Any, Protocol

from google_play_scraper import Sort, reviews, search

from review_analyzer.schemas import AppSummary, Review


class UpstreamError(RuntimeError):
    """The Play Store request failed."""


class ReviewSource(Protocol):
    def search(self, query: str, limit: int) -> list[AppSummary]: ...

    def fetch_reviews(self, app_id: str, count: int) -> list[Review]: ...


class GooglePlayClient:
    def __init__(self, language: str = "en", country: str = "us") -> None:
        self.language = language
        self.country = country

    def search(self, query: str, limit: int) -> list[AppSummary]:
        try:
            results = search(query, lang=self.language, country=self.country)
        except Exception as e:
            raise UpstreamError(f"Play Store search failed: {e}") from e
        # The "featured" top result comes back with appId=None and can't be fetched, so skip it.
        return [
            AppSummary(
                app_id=r["appId"],
                title=r["title"],
                developer=r.get("developer"),
                score=r.get("score"),
            )
            for r in results
            if r.get("appId")
        ][:limit]

    def fetch_reviews(self, app_id: str, count: int) -> list[Review]:
        try:
            raw, _ = reviews(
                app_id, lang=self.language, country=self.country, sort=Sort.NEWEST, count=count
            )
        except Exception as e:
            raise UpstreamError(f"Fetching reviews for {app_id} failed: {e}") from e
        return [to_review(app_id, r) for r in raw]


def to_review(app_id: str, raw: dict[str, Any]) -> Review:
    return Review(
        review_id=raw["reviewId"],
        app_id=app_id,
        user_name=raw.get("userName") or "",
        rating=raw["score"],
        content=raw.get("content") or "",
        created_at=raw["at"],
        app_version=raw.get("reviewCreatedVersion") or raw.get("appVersion"),
        thumbs_up=raw.get("thumbsUpCount") or 0,
    )
