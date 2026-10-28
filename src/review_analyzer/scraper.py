"""Thin wrapper around google-play-scraper that returns typed models."""

import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Protocol

from google_play_scraper import Sort, app, reviews, search
from google_play_scraper.exceptions import NotFoundError

from review_analyzer.schemas import AppSummary, Review

logger = logging.getLogger(__name__)

SEARCH_PAGE_URL = "https://play.google.com/store/search"
_DETAILS_LINK = re.compile(r"/store/apps/details\?id=([\w.]+)")
# How many unmatched app links to try when resolving the featured result.
FEATURED_CANDIDATES = 3


class UpstreamError(RuntimeError):
    """The Play Store request failed."""


class AppNotFoundError(LookupError):
    def __init__(self, app_id: str) -> None:
        super().__init__(f"No app with ID {app_id} on the Play Store.")
        self.app_id = app_id


class ReviewSource(Protocol):
    def search(self, query: str, limit: int) -> list[AppSummary]: ...

    def lookup(self, app_id: str) -> AppSummary: ...

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

        apps = []
        for result in results:
            if result.get("appId"):
                apps.append(to_summary(result))
            elif featured := self._resolve_featured(query, result["title"], results):
                apps.append(featured)
        return apps[:limit]

    def _resolve_featured(
        self, query: str, title: str, results: list[dict[str, Any]]
    ) -> AppSummary | None:
        """Find the app ID of the "featured" top result, which the scraper returns without one.

        It is the first app linked on the search page, so try the first few links
        that aren't already in the results and keep the one whose title matches.
        """
        known = {r["appId"] for r in results if r.get("appId")}
        try:
            candidates = [i for i in self._search_page_app_ids(query) if i not in known]
            for app_id in candidates[:FEATURED_CANDIDATES]:
                summary = self.lookup(app_id)
                if summary.title.casefold() == title.casefold():
                    return summary
        except (UpstreamError, AppNotFoundError, OSError) as e:
            logger.warning("Could not resolve featured result %r: %s", title, e)
        return None

    def _search_page_app_ids(self, query: str) -> list[str]:
        params = urllib.parse.urlencode(
            {"q": query, "c": "apps", "hl": self.language, "gl": self.country}
        )
        request = urllib.request.Request(
            f"{SEARCH_PAGE_URL}?{params}", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            html = response.read().decode("utf-8", errors="replace")
        return list(dict.fromkeys(_DETAILS_LINK.findall(html)))

    def lookup(self, app_id: str) -> AppSummary:
        try:
            details = app(app_id, lang=self.language, country=self.country)
        except NotFoundError as e:
            raise AppNotFoundError(app_id) from e
        except Exception as e:
            raise UpstreamError(f"Looking up {app_id} failed: {e}") from e
        return to_summary(details)

    def fetch_reviews(self, app_id: str, count: int) -> list[Review]:
        try:
            raw, _ = reviews(
                app_id, lang=self.language, country=self.country, sort=Sort.NEWEST, count=count
            )
        except Exception as e:
            raise UpstreamError(f"Fetching reviews for {app_id} failed: {e}") from e
        return [to_review(app_id, r) for r in raw]


def to_summary(raw: dict[str, Any]) -> AppSummary:
    return AppSummary(
        app_id=raw["appId"],
        title=raw["title"],
        developer=raw.get("developer"),
        score=raw.get("score"),
    )


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
