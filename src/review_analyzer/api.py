"""HTTP API: search apps, sync their reviews into the database, and serve analyses.

Run with: uvicorn review_analyzer.api:create_app --factory
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse, Response

from review_analyzer import __version__
from review_analyzer.config import Settings
from review_analyzer.ml import Embedder, SentimentModel
from review_analyzer.runtime import Runtime, build_runtime
from review_analyzer.schemas import Analysis, AppSummary, ReviewPage, SyncResult
from review_analyzer.scraper import AppNotFoundError, ReviewSource, UpstreamError
from review_analyzer.service import NoReviewsError, RateLimitedError, ReviewService

MAX_SYNC_COUNT = 1000
MAX_PAGE_SIZE = MAX_SYNC_COUNT
MAX_ANALYSIS_REVIEWS = 5000

AppId = Annotated[
    str,
    Path(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$",
        description="Android package name, e.g. com.spotify.music",
    ),
]

router = APIRouter()


def get_service(request: Request) -> Iterator[ReviewService]:
    runtime: Runtime = request.app.state.runtime
    with runtime.service() as service:
        yield service


Service = Annotated[ReviewService, Depends(get_service)]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/apps/search")
def search_apps(
    service: Service,
    q: Annotated[str, Query(min_length=1, max_length=100)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
) -> list[AppSummary]:
    return service.search(q, limit)


# Declared after /apps/search so "search" isn't taken for an app ID.
@router.get("/apps/{app_id}")
def get_app(service: Service, app_id: AppId) -> AppSummary:
    return service.lookup(app_id)


@router.post("/apps/{app_id}/sync")
def sync_reviews(
    service: Service,
    app_id: AppId,
    count: Annotated[int, Query(ge=1, le=MAX_SYNC_COUNT)] = 100,
    force: Annotated[bool, Query(description="Fetch even if recent reviews are stored.")] = False,
) -> SyncResult:
    """Fetch the newest reviews from the Play Store and store them.

    Skipped (status "fresh") when at least `count` reviews were fetched recently.
    If fetching fails but older reviews are stored, returns status "stale".
    """
    return service.sync(app_id, count, force=force)


@router.get("/apps/{app_id}/reviews")
def list_reviews(
    service: Service,
    app_id: AppId,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ReviewPage:
    return service.reviews(app_id, limit, offset)


@router.get("/apps/{app_id}/analysis")
def get_analysis(
    service: Service,
    app_id: AppId,
    limit: Annotated[int, Query(ge=1, le=MAX_ANALYSIS_REVIEWS)] = 500,
) -> Analysis:
    """Analyze the most recent `limit` stored reviews."""
    return service.analyze(app_id, limit)


@router.get("/apps/{app_id}/export.csv")
def export_reviews(
    service: Service,
    app_id: AppId,
    limit: Annotated[int, Query(ge=1, le=MAX_ANALYSIS_REVIEWS)] = 1000,
) -> Response:
    return Response(
        service.export_csv(app_id, limit),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{app_id}_reviews.csv"'},
    )


def create_app(
    settings: Settings | None = None,
    source: ReviewSource | None = None,
    sentiment_model: SentimentModel | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    runtime = build_runtime(settings, source, sentiment_model, embedder)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime.start_preload()
        yield
        runtime.engine.dispose()

    app = FastAPI(title="Review Analyzer API", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime

    @app.exception_handler(UpstreamError)
    async def upstream_error(request: Request, exc: UpstreamError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(RateLimitedError)
    async def rate_limited(request: Request, exc: RateLimitedError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"detail": str(exc)})

    @app.exception_handler(AppNotFoundError)
    @app.exception_handler(NoReviewsError)
    async def not_found(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(router)
    return app
