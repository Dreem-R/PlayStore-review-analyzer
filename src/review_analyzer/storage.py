"""SQLite persistence for fetched reviews, so repeat analyses don't hit the Play Store."""

from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, Index, Text, create_engine, func, make_url, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

from review_analyzer.schemas import Review


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
    fetched_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


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
    return create_engine(url, connect_args=connect_args)


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(engine)


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
                row.rating = review.rating
                row.content = review.content
                row.app_version = review.app_version
                row.thumbs_up = review.thumbs_up
        self._session.commit()
        return len(incoming) - len(existing)

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
