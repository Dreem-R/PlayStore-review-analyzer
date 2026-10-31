"""Where the UI gets its data: a running API over HTTP, or the analysis code in-process.

The in-process backend exists for hosts that can only run a single Streamlit
process, such as Streamlit Community Cloud. Set REVIEW_ANALYZER_API_URL to use an API.
"""

import os
from collections.abc import Callable
from typing import Any, Literal, Protocol, TypeVar

import httpx

from review_analyzer.runtime import Runtime, build_runtime
from review_analyzer.scraper import AppNotFoundError, UpstreamError
from review_analyzer.service import NoReviewsError, RateLimitedError, ReviewService

T = TypeVar("T")
ErrorKind = Literal["unavailable", "not_found", "other"]
ERROR_KINDS: dict[int, ErrorKind] = {502: "unavailable", 429: "unavailable", 404: "not_found"}


class BackendError(Exception):
    def __init__(self, message: str, kind: ErrorKind = "other") -> None:
        super().__init__(message)
        self.kind = kind


class Backend(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...

    def lookup(self, app_id: str) -> dict[str, Any]: ...

    def sync(self, app_id: str, count: int) -> dict[str, Any]: ...

    def analysis(self, app_id: str, limit: int) -> dict[str, Any]: ...

    def reviews(self, app_id: str, limit: int) -> list[dict[str, Any]]: ...

    def export_csv(self, app_id: str, limit: int) -> str: ...


class HttpBackend:
    def __init__(self, base_url: str, client: httpx.Client | None = None) -> None:
        self.base_url = base_url
        # Long read timeout: the first analysis of an app scores every review.
        self._client = client or httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(10.0, read=900.0)
        )

    def _call(self, method: str, path: str, **params: Any) -> Any:
        try:
            response = self._client.request(method, path, params=params)
        except httpx.HTTPError as e:
            raise BackendError(f"Can't reach the API at {self.base_url} ({e}).") from e
        if response.is_error:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            kind = ERROR_KINDS.get(response.status_code, "other")
            raise BackendError(str(detail), kind)
        if "json" in response.headers.get("content-type", ""):
            return response.json()
        return response.text

    def search(self, query: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = self._call("GET", "/apps/search", q=query)
        return result

    def lookup(self, app_id: str) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", f"/apps/{app_id}")
        return result

    def sync(self, app_id: str, count: int) -> dict[str, Any]:
        result: dict[str, Any] = self._call("POST", f"/apps/{app_id}/sync", count=count)
        return result

    def analysis(self, app_id: str, limit: int) -> dict[str, Any]:
        result: dict[str, Any] = self._call("GET", f"/apps/{app_id}/analysis", limit=limit)
        return result

    def reviews(self, app_id: str, limit: int) -> list[dict[str, Any]]:
        page = self._call("GET", f"/apps/{app_id}/reviews", limit=limit)
        items: list[dict[str, Any]] = page["items"]
        return items

    def export_csv(self, app_id: str, limit: int) -> str:
        return str(self._call("GET", f"/apps/{app_id}/export.csv", limit=limit))


class LocalBackend:
    """Runs ReviewService in this process, with the same storage and caching as the API."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def _call(self, action: Callable[[ReviewService], T]) -> T:
        try:
            with self.runtime.service() as service:
                return action(service)
        except (UpstreamError, RateLimitedError) as e:
            raise BackendError(str(e), "unavailable") from e
        except (AppNotFoundError, NoReviewsError) as e:
            raise BackendError(str(e), "not_found") from e

    def search(self, query: str) -> list[dict[str, Any]]:
        return self._call(lambda s: [a.model_dump(mode="json") for a in s.search(query, 5)])

    def lookup(self, app_id: str) -> dict[str, Any]:
        return self._call(lambda s: s.lookup(app_id).model_dump(mode="json"))

    def sync(self, app_id: str, count: int) -> dict[str, Any]:
        return self._call(lambda s: s.sync(app_id, count).model_dump(mode="json"))

    def analysis(self, app_id: str, limit: int) -> dict[str, Any]:
        return self._call(lambda s: s.analyze(app_id, limit).model_dump(mode="json"))

    def reviews(self, app_id: str, limit: int) -> list[dict[str, Any]]:
        return self._call(
            lambda s: [r.model_dump(mode="json") for r in s.reviews(app_id, limit, 0).items]
        )

    def export_csv(self, app_id: str, limit: int) -> str:
        return self._call(lambda s: s.export_csv(app_id, limit))


def create_backend() -> Backend:
    api_url = os.environ.get("REVIEW_ANALYZER_API_URL")
    if api_url:
        return HttpBackend(api_url)
    runtime = build_runtime()
    runtime.start_preload()
    return LocalBackend(runtime)
