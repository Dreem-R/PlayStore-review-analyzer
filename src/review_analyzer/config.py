from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from REVIEW_ANALYZER_* environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_ANALYZER_", env_file=".env", extra="ignore"
    )

    database_url: str = "sqlite:///data/reviews.db"
    language: str = "en"
    country: str = "us"

    # Fetching from the Play Store. The public demo tightens these (see deploy/streamlit).
    sync_ttl_minutes: int = 360
    max_sync_count: int = 1000
    max_syncs_per_hour: int | None = None
    # Apps to fetch and analyze in the background at startup, so they're instant to open.
    preload_apps: list[str] = []
    preload_count: int = 200
    # Models are only used when the `ml` extra is installed.
    enable_ml: bool = True
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
