"""HTTP API: search apps, sync their reviews into the database, and serve analyses.

Run with: uvicorn review_analyzer.api:create_app --factory
"""

import csv
import io
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, Path, Query, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import sessionmaker

from review_analyzer import __version__
from review_analyzer.analysis import Summarizer
from review_analyzer.config import Settings
from review_analyzer.schemas import Analysis, AppSummary, ReviewPage, SyncResult
from review_analyzer.scraper import GooglePlayClient, ReviewSource, UpstreamError
from review_analyzer.service import NoReviewsError, ReviewService
from review_analyzer.storage import ReviewRepository, init_db, make_engine
from review_analyzer.summarizer import LazySummarizer

MAX_SYNC_COUNT = 1000
MAX_PAGE_SIZE = 500
MAX_ANALYSIS_REVIEWS = 5000
CSV_COLUMNS = (
    "review_id",
    "user_name",
    "rating",
    "created_at",
    "app_version",
    "thumbs_up",
    "content",
)

AppId = Annotated[
    str,
    Path(
        pattern=r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)+$",
        description="Android package name, e.g. com.spotify.music",
    ),
]

router = APIRouter()


def get_service(request: Request) -> Iterator[ReviewService]:
    state = request.app.state
    with state.session_factory() as session:
        yield ReviewService(state.source, ReviewRepository(session), state.summarizer)


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


@router.post("/apps/{app_id}/sync")
def sync_reviews(
    service: Service,
    app_id: AppId,
    count: Annotated[int, Query(ge=1, le=MAX_SYNC_COUNT)] = 100,
) -> SyncResult:
    """Fetch the newest reviews from the Play Store and store them."""
    return service.sync(app_id, count)


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
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for review in service.reviews(app_id, limit, 0).items:
        writer.writerow(review.model_dump())
    return Response(
        buffer.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{app_id}_reviews.csv"'},
    )


def create_app(
    settings: Settings | None = None,
    source: ReviewSource | None = None,
    summarizer: Summarizer | None = None,
) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url)
    if summarizer is None and settings.enable_summarizer and LazySummarizer.available():
        summarizer = LazySummarizer(settings.summarizer_model)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        init_db(engine)
        yield
        engine.dispose()

    app = FastAPI(title="Review Analyzer API", version=__version__, lifespan=lifespan)
    app.state.session_factory = sessionmaker(engine, expire_on_commit=False)
    app.state.source = source or GooglePlayClient(settings.language, settings.country)
    app.state.summarizer = summarizer

    @app.exception_handler(UpstreamError)
    async def upstream_error(request: Request, exc: UpstreamError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(NoReviewsError)
    async def no_reviews(request: Request, exc: NoReviewsError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    app.include_router(router)
    return app
