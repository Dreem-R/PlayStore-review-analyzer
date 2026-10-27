from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from REVIEW_ANALYZER_* environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_ANALYZER_", env_file=".env", extra="ignore"
    )

    database_url: str = "sqlite:///data/reviews.db"
    language: str = "en"
    country: str = "us"
    summarizer_model: str = "sshleifer/distilbart-cnn-6-6"
    enable_summarizer: bool = True
