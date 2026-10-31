import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from review_analyzer.api import create_app
from review_analyzer.config import Settings
from review_analyzer.schemas import AppSummary, Review, Sentiment
from review_analyzer.scraper import AppNotFoundError, UpstreamError
from review_analyzer.storage import ReviewRepository, init_db, make_engine

APP_ID = "com.example.notes"
FIXTURES = Path(__file__).parent / "fixtures"


class FakeSource:
    """Stands in for the Play Store so tests never touch the network."""

    def __init__(self, reviews: list[Review]) -> None:
        self.reviews = reviews
        self.fail = False
        self.fetches: list[tuple[str, int]] = []

    def search(self, query: str, limit: int) -> list[AppSummary]:
        if self.fail:
            raise UpstreamError("Play Store search failed: timeout")
        return [AppSummary(app_id=APP_ID, title="Example Notes", developer="Example Inc.")][:limit]

    def lookup(self, app_id: str) -> AppSummary:
        if app_id != APP_ID:
            raise AppNotFoundError(app_id)
        return AppSummary(app_id=APP_ID, title="Example Notes")

    def fetch_reviews(self, app_id: str, count: int) -> list[Review]:
        self.fetches.append((app_id, count))
        if self.fail:
            raise UpstreamError(f"Fetching reviews for {app_id} failed: timeout")
        return [r for r in self.reviews if r.app_id == app_id][:count]


class FakeSentiment:
    """Keyword rules instead of a transformer. Records every batch it scores."""

    model_name = "fake-sentiment"

    NEGATIVE = ("crash", "ads", " ad ", "can't", "slow", "lost", "terrible", "nobody")
    POSITIVE = ("love", "great", "clean", "good", "never had a problem")

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def predict(self, texts: Sequence[str]) -> list[tuple[Sentiment, float]]:
        self.batches.append(list(texts))
        return [self._label(f" {t.lower()} ") for t in texts]

    def _label(self, text: str) -> tuple[Sentiment, float]:
        if any(word in text for word in self.NEGATIVE):
            return "negative", 0.9
        if any(word in text for word in self.POSITIVE):
            return "positive", 0.9
        return "neutral", 0.6


class FakeEmbedder:
    """Puts sentences about the same topic on the same axis, so clustering is predictable.

    Vectors are fixed-width so they can be cached and stacked across calls.
    """

    model_name = "fake-embedder"
    WIDTH = 64

    TOPICS = (("crash", "slow", "freez"), ("ads", " ad "), ("log in", "verification"))

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self.calls += len(texts)
        vectors = np.zeros((len(texts), self.WIDTH), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f" {text.lower()} "
            topic = next(
                (i for i, words in enumerate(self.TOPICS) if any(w in padded for w in words)),
                None,
            )
            # Off-topic sentences get an axis from their text, so they don't cluster together.
            axis = topic if topic is not None else 3 + sum(map(ord, text)) % (self.WIDTH - 3)
            vectors[row, axis] = 1.0
        return vectors


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
def sentiment_model() -> FakeSentiment:
    return FakeSentiment()


def _client(source: FakeSource, **models: object) -> Iterator[TestClient]:
    settings = Settings(database_url="sqlite://", enable_ml=False)
    with TestClient(create_app(settings, source=source, **models)) as client:  # type: ignore[arg-type]
        yield client


@pytest.fixture
def client(source: FakeSource) -> Iterator[TestClient]:
    """API without models: keyword issues, rating-only sentiment."""
    yield from _client(source)


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def ml_client(
    source: FakeSource, sentiment_model: FakeSentiment, embedder: FakeEmbedder
) -> Iterator[TestClient]:
    """API with fake models: text sentiment and clustering."""
    yield from _client(source, sentiment_model=sentiment_model, embedder=embedder)
