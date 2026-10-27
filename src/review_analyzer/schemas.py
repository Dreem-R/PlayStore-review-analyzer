from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


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


class ReviewPage(BaseModel):
    items: list[Review]
    total: int
    limit: int
    offset: int


class SyncResult(BaseModel):
    app_id: str
    fetched: int
    new: int
    total_stored: int


class SentimentCounts(BaseModel):
    """Derived from the star rating: 4-5 positive, 3 neutral, 1-2 negative."""

    positive: int = 0
    neutral: int = 0
    negative: int = 0


class IssueCategory(BaseModel):
    name: str
    mentions: int
    suggestion: str


class Analysis(BaseModel):
    app_id: str
    review_count: int
    sentiment: SentimentCounts
    issues: list[IssueCategory] = Field(
        description="Categories found in negative reviews, most mentioned first."
    )
    examples: list[str] = Field(
        description="Sentences from negative reviews that matched a category."
    )
    summary: str | None = Field(
        default=None,
        description="Model-written summary of negative reviews, only when no category matched.",
    )
