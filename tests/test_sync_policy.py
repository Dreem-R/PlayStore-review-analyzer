from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
from sqlalchemy.orm import Session

from review_analyzer.schemas import Review
from review_analyzer.scraper import UpstreamError
from review_analyzer.service import RateLimitedError, ReviewService, SyncPolicy
from review_analyzer.storage import ReviewRepository, SyncLog, init_db, make_engine
from tests.conftest import APP_ID, FakeSource

START = datetime(2026, 10, 31, 12, 0)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def session() -> Iterator[Session]:
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as session:
        yield session


def make_service(
    session: Session, source: FakeSource, clock: Clock, **policy: object
) -> ReviewService:
    return ReviewService(
        source,
        ReviewRepository(session),
        SyncLog(session),
        SyncPolicy(**policy),  # type: ignore[arg-type]
        clock=clock,
    )


def test_recent_sync_is_reused(session: Session, source: FakeSource, clock: Clock) -> None:
    service = make_service(session, source, clock, ttl=timedelta(hours=6))
    first = service.sync(APP_ID, 10)
    clock.advance(hours=5)

    second = service.sync(APP_ID, 10)

    assert first.status == "fetched"
    assert second.status == "fresh"
    assert second.total_stored == 10
    assert second.last_synced_at == START
    assert len(source.fetches) == 1


def test_sync_fetches_again_after_ttl(session: Session, source: FakeSource, clock: Clock) -> None:
    service = make_service(session, source, clock, ttl=timedelta(hours=6))
    service.sync(APP_ID, 10)
    clock.advance(hours=6)

    assert service.sync(APP_ID, 10).status == "fetched"
    assert len(source.fetches) == 2


def test_asking_for_more_reviews_fetches_again(
    session: Session, source: FakeSource, clock: Clock
) -> None:
    service = make_service(session, source, clock)
    service.sync(APP_ID, 5)

    assert service.sync(APP_ID, 10).status == "fetched"
    assert service.sync(APP_ID, 5).status == "fresh"  # 10 were fetched, which covers 5


def test_force_bypasses_ttl(session: Session, source: FakeSource, clock: Clock) -> None:
    service = make_service(session, source, clock)
    service.sync(APP_ID, 10)
    assert service.sync(APP_ID, 10, force=True).status == "fetched"


def test_count_is_capped(session: Session, source: FakeSource, clock: Clock) -> None:
    service = make_service(session, source, clock, max_count=3)
    assert service.sync(APP_ID, 10).fetched == 3
    assert source.fetches == [(APP_ID, 3)]


def test_rate_limit_without_stored_reviews_raises(
    session: Session, source: FakeSource, clock: Clock, reviews: list[Review]
) -> None:
    other = [
        r.model_copy(update={"app_id": "com.other.app", "review_id": f"o{i}"})
        for i, r in enumerate(reviews)
    ]
    source.reviews += other
    service = make_service(session, source, clock, max_per_hour=1)
    service.sync(APP_ID, 10)

    with pytest.raises(RateLimitedError):
        service.sync("com.other.app", 10)


def test_rate_limit_with_stored_reviews_is_stale(
    session: Session, source: FakeSource, clock: Clock
) -> None:
    service = make_service(session, source, clock, max_per_hour=1, ttl=timedelta(minutes=1))
    service.sync(APP_ID, 10)
    clock.advance(minutes=5)

    result = service.sync(APP_ID, 10)

    assert result.status == "stale"
    assert result.total_stored == 10
    assert result.last_synced_at == START
    assert result.message is not None and "limit" in result.message
    assert len(source.fetches) == 1


def test_rate_limit_window_slides(session: Session, source: FakeSource, clock: Clock) -> None:
    service = make_service(session, source, clock, max_per_hour=1, ttl=timedelta(minutes=1))
    service.sync(APP_ID, 10)
    clock.advance(minutes=61)
    assert service.sync(APP_ID, 10).status == "fetched"


def test_upstream_failure_with_stored_reviews_is_stale(
    session: Session, source: FakeSource, clock: Clock
) -> None:
    service = make_service(session, source, clock)
    service.sync(APP_ID, 10)
    source.fail = True

    result = service.sync(APP_ID, 10, force=True)

    assert result.status == "stale"
    assert result.message is not None and "timeout" in result.message


def test_upstream_failure_without_stored_reviews_raises(
    session: Session, source: FakeSource, clock: Clock
) -> None:
    source.fail = True
    with pytest.raises(UpstreamError):
        make_service(session, source, clock).sync(APP_ID, 10)


def test_failed_fetches_are_not_logged(session: Session, source: FakeSource, clock: Clock) -> None:
    source.fail = True
    with pytest.raises(UpstreamError):
        make_service(session, source, clock).sync(APP_ID, 10)
    assert SyncLog(session).last(APP_ID) is None
