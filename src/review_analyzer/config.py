from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from REVIEW_ANALYZER_* environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="REVIEW_ANALYZER_", env_file=".env", extra="ignore"
    )

    database_url: str = "sqlite:///data/reviews.db"
    language: str = "en"
    country: str = "us"
    # Models are only used when the `ml` extra is installed.
    enable_ml: bool = True
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
