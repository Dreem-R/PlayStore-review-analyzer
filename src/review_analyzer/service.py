import csv
import io
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from review_analyzer.analysis import analyze
from review_analyzer.ml import Embedder, SentimentModel
from review_analyzer.schemas import Analysis, AppSummary, Review, ReviewPage, SyncResult
from review_analyzer.scraper import ReviewSource, UpstreamError
from review_analyzer.storage import ReviewRepository, SyncLog, utcnow

CSV_COLUMNS = (
    "review_id",
    "user_name",
    "rating",
    "created_at",
    "app_version",
    "thumbs_up",
    "text_sentiment",
    "sentiment_score",
    "content",
)


class NoReviewsError(LookupError):
    def __init__(self, app_id: str) -> None:
        super().__init__(f"No reviews stored for {app_id}. Sync the app first.")
        self.app_id = app_id


class RateLimitedError(RuntimeError):
    """Too many Play Store fetches in the last hour."""


@dataclass(frozen=True)
class SyncPolicy:
    """How eagerly to fetch from the Play Store.

    The scraper is unofficial and Google can rate-limit it, so a public deployment
    re-uses recent fetches and caps how often it fetches at all.
    """

    ttl: timedelta = timedelta(hours=6)
    max_count: int = 1000
    max_per_hour: int | None = None


class ReviewService:
    """Fetches reviews from the store, persists them, and analyzes what's stored."""

    def __init__(
        self,
        source: ReviewSource,
        repository: ReviewRepository,
        sync_log: SyncLog,
        policy: SyncPolicy | None = None,
        sentiment_model: SentimentModel | None = None,
        embedder: Embedder | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        self._source = source
        self._repository = repository
        self._sync_log = sync_log
        self._policy = policy or SyncPolicy()
        self._sentiment_model = sentiment_model
        self._embedder = embedder
        self._clock = clock

    def search(self, query: str, limit: int) -> list[AppSummary]:
        return self._source.search(query, limit)

    def lookup(self, app_id: str) -> AppSummary:
        return self._source.lookup(app_id)

    def sync(self, app_id: str, count: int, force: bool = False) -> SyncResult:
        """Fetch the newest reviews, unless recent enough ones are stored already.

        If the fetch fails or is rate-limited but older reviews are stored, returns
        them as "stale" instead of failing, so the caller can still analyze them.
        """
        count = min(count, self._policy.max_count)
        now = self._clock()
        last = self._sync_log.last(app_id)
        stored = self._repository.count(app_id)
        last_synced_at = last.synced_at if last else None

        if (
            not force
            and last is not None
            and last.requested >= count
            and now - last.synced_at < self._policy.ttl
        ):
            return self._unchanged(app_id, "fresh", stored, last_synced_at)

        try:
            self._check_rate_limit(now)
            reviews = self._source.fetch_reviews(app_id, count)
        except (UpstreamError, RateLimitedError) as e:
            if stored == 0:
                raise
            return self._unchanged(app_id, "stale", stored, last_synced_at, message=str(e))

        new = self._repository.upsert(reviews)
        self._sync_log.record(app_id, requested=count, fetched=len(reviews), at=now)
        return SyncResult(
            app_id=app_id,
            status="fetched",
            fetched=len(reviews),
            new=new,
            total_stored=self._repository.count(app_id),
            last_synced_at=now,
        )

    def _check_rate_limit(self, now: datetime) -> None:
        limit = self._policy.max_per_hour
        if limit is not None and self._sync_log.count_since(now - timedelta(hours=1)) >= limit:
            raise RateLimitedError(f"Play Store fetch limit reached ({limit} per hour).")

    @staticmethod
    def _unchanged(
        app_id: str,
        status: Literal["fresh", "stale"],
        stored: int,
        last_synced_at: datetime | None,
        message: str | None = None,
    ) -> SyncResult:
        return SyncResult(
            app_id=app_id,
            status=status,
            fetched=0,
            new=0,
            total_stored=stored,
            last_synced_at=last_synced_at,
            message=message,
        )

    def reviews(self, app_id: str, limit: int, offset: int) -> ReviewPage:
        return ReviewPage(
            items=self._repository.latest(app_id, limit, offset),
            total=self._repository.count(app_id),
            limit=limit,
            offset=offset,
        )

    def export_csv(self, app_id: str, limit: int) -> str:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for review in self._repository.latest(app_id, limit):
            writer.writerow(review.model_dump())
        return buffer.getvalue()

    def analyze(self, app_id: str, limit: int) -> Analysis:
        reviews = self._repository.latest(app_id, limit)
        if not reviews:
            raise NoReviewsError(app_id)
        reviews = self._score_sentiment(reviews)
        return analyze(app_id, reviews, self._embedder, self._sentiment_model)

    def _score_sentiment(self, reviews: list[Review]) -> list[Review]:
        """Score reviews that have no text sentiment yet and persist the result.

        Scoring happens here rather than at sync time so syncs stay fast and
        reviews stored before the model was installed get scored too.
        """
        unscored = [r for r in reviews if r.text_sentiment is None]
        if self._sentiment_model is None or not unscored:
            return reviews
        predictions = dict(
            zip(
                (r.review_id for r in unscored),
                self._sentiment_model.predict([r.content for r in unscored]),
                strict=True,
            )
        )
        self._repository.save_sentiment(predictions)
        return [
            r.model_copy(
                update={
                    "text_sentiment": predictions[r.review_id][0],
                    "sentiment_score": predictions[r.review_id][1],
                }
            )
            if r.review_id in predictions
            else r
            for r in reviews
        ]
