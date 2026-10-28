from review_analyzer.analysis import analyze
from review_analyzer.ml import Embedder, SentimentModel
from review_analyzer.schemas import Analysis, AppSummary, Review, ReviewPage, SyncResult
from review_analyzer.scraper import ReviewSource
from review_analyzer.storage import ReviewRepository


class NoReviewsError(LookupError):
    def __init__(self, app_id: str) -> None:
        super().__init__(f"No reviews stored for {app_id}. Sync the app first.")
        self.app_id = app_id


class ReviewService:
    """Fetches reviews from the store, persists them, and analyzes what's stored."""

    def __init__(
        self,
        source: ReviewSource,
        repository: ReviewRepository,
        sentiment_model: SentimentModel | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self._source = source
        self._repository = repository
        self._sentiment_model = sentiment_model
        self._embedder = embedder

    def search(self, query: str, limit: int) -> list[AppSummary]:
        return self._source.search(query, limit)

    def lookup(self, app_id: str) -> AppSummary:
        return self._source.lookup(app_id)

    def sync(self, app_id: str, count: int) -> SyncResult:
        reviews = self._source.fetch_reviews(app_id, count)
        new = self._repository.upsert(reviews)
        return SyncResult(
            app_id=app_id,
            fetched=len(reviews),
            new=new,
            total_stored=self._repository.count(app_id),
        )

    def reviews(self, app_id: str, limit: int, offset: int) -> ReviewPage:
        return ReviewPage(
            items=self._repository.latest(app_id, limit, offset),
            total=self._repository.count(app_id),
            limit=limit,
            offset=offset,
        )

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
