"""SQLite persistence for fetched reviews, so repeat analyses don't hit the Play Store."""

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from alembic import command
from alembic.config import Config
from numpy.typing import NDArray
from sqlalchemy import (
    Engine,
    Float,
    Index,
    LargeBinary,
    String,
    Text,
    create_engine,
    event,
    func,
    inspect,
    make_url,
    select,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from review_analyzer.schemas import Review, Sentiment

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# Schema that databases created before migrations were introduced already have.
BASELINE_REVISION = "0001"


def utcnow() -> datetime:
    """Naive UTC, which is what SQLite stores and returns."""
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class ReviewRow(Base):
    __tablename__ = "reviews"
    __table_args__ = (Index("ix_reviews_app_created", "app_id", "created_at"),)

    review_id: Mapped[str] = mapped_column(primary_key=True)
    app_id: Mapped[str]
    user_name: Mapped[str]
    rating: Mapped[int]
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime]
    app_version: Mapped[str | None]
    thumbs_up: Mapped[int] = mapped_column(default=0)
    fetched_at: Mapped[datetime] = mapped_column(default=utcnow)
    text_sentiment: Mapped[str | None]
    sentiment_score: Mapped[float | None] = mapped_column(Float)


class SentimentCacheRow(Base):
    __tablename__ = "sentiment_cache"

    model: Mapped[str] = mapped_column(primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    label: Mapped[str]
    score: Mapped[float] = mapped_column(Float)


class EmbeddingCacheRow(Base):
    __tablename__ = "embedding_cache"

    model: Mapped[str] = mapped_column(primary_key=True)
    text_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    vector: Mapped[bytes] = mapped_column(LargeBinary)


class SyncRow(Base):
    """One fetch from the Play Store. Drives freshness checks and the hourly fetch limit."""

    __tablename__ = "syncs"
    __table_args__ = (Index("ix_syncs_app_synced", "app_id", "synced_at"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    app_id: Mapped[str]
    synced_at: Mapped[datetime] = mapped_column(index=True)
    requested: Mapped[int]
    fetched: Mapped[int]


@dataclass(frozen=True)
class SyncRecord:
    app_id: str
    synced_at: datetime
    requested: int
    fetched: int


def make_engine(database_url: str) -> Engine:
    url = make_url(database_url)
    if url.get_backend_name() != "sqlite":
        return create_engine(url)

    # FastAPI runs sync endpoints in a thread pool, so connections cross threads.
    connect_args = {"check_same_thread": False}
    if url.database in (None, "", ":memory:"):
        # One shared connection, otherwise every connection gets its own empty in-memory DB.
        return create_engine(url, connect_args=connect_args, poolclass=StaticPool)
    Path(url.database).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args=connect_args)

    @event.listens_for(engine, "connect")
    def configure(connection: Any, record: Any) -> None:
        # WAL lets the background preload write while visitors read, and the busy
        # timeout makes concurrent writers wait instead of failing with "database is locked".
        cursor = connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    return engine


def init_db(engine: Engine) -> None:
    """Apply pending migrations."""
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR))
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        tables = inspect(connection).get_table_names()
        if "reviews" in tables and "alembic_version" not in tables:
            command.stamp(config, BASELINE_REVISION)
        command.upgrade(config, "head")


class ReviewRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, reviews: Iterable[Review]) -> int:
        """Insert new reviews and refresh edited ones. Returns how many were new."""
        incoming = {r.review_id: r for r in reviews}
        if not incoming:
            return 0

        existing = {
            row.review_id: row
            for row in self._session.scalars(
                select(ReviewRow).where(ReviewRow.review_id.in_(incoming))
            )
        }
        for review_id, review in incoming.items():
            row = existing.get(review_id)
            if row is None:
                self._session.add(ReviewRow(**review.model_dump()))
            else:
                # Users can edit a review after posting it.
                if row.content != review.content:
                    row.text_sentiment = None
                    row.sentiment_score = None
                row.rating = review.rating
                row.content = review.content
                row.app_version = review.app_version
                row.thumbs_up = review.thumbs_up
        self._session.commit()
        return len(incoming) - len(existing)

    def save_sentiment(self, predictions: dict[str, tuple[Sentiment, float]]) -> None:
        for review_id, (label, score) in predictions.items():
            self._session.execute(
                update(ReviewRow)
                .where(ReviewRow.review_id == review_id)
                .values(text_sentiment=label, sentiment_score=score)
            )
        self._session.commit()

    def count(self, app_id: str) -> int:
        stmt = select(func.count()).select_from(ReviewRow).where(ReviewRow.app_id == app_id)
        return self._session.scalar(stmt) or 0

    def latest(self, app_id: str, limit: int, offset: int = 0) -> list[Review]:
        stmt = (
            select(ReviewRow)
            .where(ReviewRow.app_id == app_id)
            .order_by(ReviewRow.created_at.desc(), ReviewRow.review_id)
            .limit(limit)
            .offset(offset)
        )
        return [Review.model_validate(row) for row in self._session.scalars(stmt)]


class SyncLog:
    def __init__(self, session: Session) -> None:
        self._session = session

    def record(self, app_id: str, requested: int, fetched: int, at: datetime) -> None:
        self._session.add(
            SyncRow(app_id=app_id, synced_at=at, requested=requested, fetched=fetched)
        )
        self._session.commit()

    def last(self, app_id: str) -> SyncRecord | None:
        row = self._session.scalars(
            select(SyncRow).where(SyncRow.app_id == app_id).order_by(SyncRow.synced_at.desc())
        ).first()
        if row is None:
            return None
        return SyncRecord(row.app_id, row.synced_at, row.requested, row.fetched)

    def count_since(self, since: datetime) -> int:
        stmt = select(func.count()).select_from(SyncRow).where(SyncRow.synced_at >= since)
        return self._session.scalar(stmt) or 0


def _chunks(items: Sequence[str], size: int = 500) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ModelOutputCache:
    """Model outputs keyed by (model name, SHA-256 of the input text)."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def sentiments(self, model: str, hashes: Collection[str]) -> dict[str, tuple[Sentiment, float]]:
        found: dict[str, tuple[Sentiment, float]] = {}
        for chunk in _chunks(sorted(hashes)):
            rows = self._session.scalars(
                select(SentimentCacheRow).where(
                    SentimentCacheRow.model == model, SentimentCacheRow.text_hash.in_(chunk)
                )
            )
            found |= {row.text_hash: (cast(Sentiment, row.label), row.score) for row in rows}
        return found

    def save_sentiments(self, model: str, outputs: dict[str, tuple[Sentiment, float]]) -> None:
        self._insert_ignoring_duplicates(
            SentimentCacheRow.__table__,
            [
                {"model": model, "text_hash": h, "label": label, "score": score}
                for h, (label, score) in outputs.items()
            ],
        )

    def embeddings(self, model: str, hashes: Collection[str]) -> dict[str, NDArray[np.float32]]:
        found: dict[str, NDArray[np.float32]] = {}
        for chunk in _chunks(sorted(hashes)):
            rows = self._session.scalars(
                select(EmbeddingCacheRow).where(
                    EmbeddingCacheRow.model == model, EmbeddingCacheRow.text_hash.in_(chunk)
                )
            )
            found |= {row.text_hash: np.frombuffer(row.vector, dtype=np.float32) for row in rows}
        return found

    def save_embeddings(self, model: str, outputs: dict[str, NDArray[np.float32]]) -> None:
        self._insert_ignoring_duplicates(
            EmbeddingCacheRow.__table__,
            [
                {"model": model, "text_hash": h, "vector": v.astype(np.float32).tobytes()}
                for h, v in outputs.items()
            ],
        )

    def _insert_ignoring_duplicates(self, table: Any, rows: list[dict[str, Any]]) -> None:
        # Two concurrent analyses can compute the same text; the second insert is a no-op.
        if not rows:
            return
        dialect = self._session.get_bind().dialect.name
        statement: Any
        if dialect == "sqlite":
            statement = sqlite.insert(table)
        elif dialect == "postgresql":
            statement = postgresql.insert(table)
        else:
            raise NotImplementedError(f"Model output cache doesn't support {dialect}")
        self._session.execute(statement.on_conflict_do_nothing(), rows)
        self._session.commit()
