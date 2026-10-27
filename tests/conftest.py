import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from review_analyzer.api import create_app
from review_analyzer.config import Settings
from review_analyzer.schemas import AppSummary, Review
from review_analyzer.scraper import UpstreamError
from review_analyzer.storage import ReviewRepository, init_db, make_engine

APP_ID = "com.example.notes"
FIXTURES = Path(__file__).parent / "fixtures"


class FakeSource:
    """Stands in for the Play Store so tests never touch the network."""

    def __init__(self, reviews: list[Review]) -> None:
        self.reviews = reviews
        self.fail = False

    def search(self, query: str, limit: int) -> list[AppSummary]:
        if self.fail:
            raise UpstreamError("Play Store search failed: timeout")
        return [AppSummary(app_id=APP_ID, title="Example Notes", developer="Example Inc.")][:limit]

    def fetch_reviews(self, app_id: str, count: int) -> list[Review]:
        if self.fail:
            raise UpstreamError(f"Fetching reviews for {app_id} failed: timeout")
        return [r for r in self.reviews if r.app_id == app_id][:count]


@pytest.fixture
def reviews() -> list[Review]:
    raw = json.loads((FIXTURES / "reviews.json").read_text(encoding="utf-8"))
    return [Review.model_validate(r) for r in raw]


@pytest.fixture
def repository() -> Iterator[ReviewRepository]:
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as session:
        yield ReviewRepository(session)


@pytest.fixture
def source(reviews: list[Review]) -> FakeSource:
    return FakeSource(reviews)


@pytest.fixture
def client(source: FakeSource) -> Iterator[TestClient]:
    app = create_app(Settings(database_url="sqlite://", enable_summarizer=False), source=source)
    with TestClient(app) as client:
        yield client
