from review_analyzer.analysis import Summarizer, analyze
from review_analyzer.schemas import Analysis, AppSummary, ReviewPage, SyncResult
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
        summarizer: Summarizer | None = None,
    ) -> None:
        self._source = source
        self._repository = repository
        self._summarizer = summarizer

    def search(self, query: str, limit: int) -> list[AppSummary]:
        return self._source.search(query, limit)

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
        return analyze(app_id, reviews, self._summarizer)
