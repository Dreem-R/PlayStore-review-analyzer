from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["positive", "neutral", "negative"]


class AppSummary(BaseModel):
    app_id: str
    title: str
    developer: str | None = None
    score: float | None = None


class Review(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    review_id: str
    app_id: str
    user_name: str
    rating: int = Field(ge=1, le=5)
    content: str
    created_at: datetime
    app_version: str | None = None
    thumbs_up: int = 0
    text_sentiment: Sentiment | None = Field(
        default=None, description="Predicted from the review text; null until scored."
    )
    sentiment_score: float | None = Field(default=None, description="Model confidence, 0-1.")


class ReviewPage(BaseModel):
    items: list[Review]
    total: int
    limit: int
    offset: int


class SyncResult(BaseModel):
    app_id: str
    status: Literal["fetched", "fresh", "stale"] = Field(
        description=(
            "fetched: reviews were just fetched. fresh: the stored reviews are recent enough, "
            "so nothing was fetched. stale: fetching failed or was rate-limited, so the "
            "stored reviews are older than usual."
        )
    )
    fetched: int
    new: int
    total_stored: int
    last_synced_at: datetime | None = Field(
        default=None, description="When reviews were last fetched for this app (UTC)."
    )
    message: str | None = Field(default=None, description="Why a fetch was skipped or failed.")


class SentimentCounts(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class IssueReview(BaseModel):
    review_id: str
    sentences: list[str] = Field(description="The sentences of this review that raised the issue.")


class Issue(BaseModel):
    label: str = Field(description="Most distinctive terms in this group of complaints.")
    category: str | None = Field(
        default=None, description="Known issue category most of the complaints fall under."
    )
    suggestion: str | None = None
    reviews: int = Field(description="Distinct reviews raising this issue.")
    share: float = Field(description="Fraction of complaint reviews raising this issue.")
    examples: list[str]
    by_version: dict[str, int] = Field(
        description="Reviews raising this issue, per app version they were written on."
    )
    members: list[IssueReview] = Field(
        description="Every review raising this issue, most helpful first, with matching sentences."
    )


class VersionStats(BaseModel):
    version: str
    reviews: int
    average_rating: float
    negative_share: float


class Analysis(BaseModel):
    app_id: str
    review_count: int
    rating_sentiment: SentimentCounts = Field(
        description="From the star rating: 4-5 positive, 3 neutral, 1-2 negative."
    )
    text_sentiment: SentimentCounts | None = Field(
        default=None, description="From the review text; null when no sentiment model is loaded."
    )
    complaint_count: int = Field(
        description="Reviews rated 1-2 stars or whose text reads as negative."
    )
    issue_method: Literal["clustering", "keywords"]
    issues: list[Issue] = Field(description="Recurring complaints, most common first.")
    mismatches: list[Review] = Field(
        description="4-5 star reviews whose text is confidently negative."
    )
    versions: list[VersionStats] = Field(description="Per-version stats, oldest first.")
