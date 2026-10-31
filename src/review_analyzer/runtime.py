"""Wires settings, storage, the Play Store client and the models into ReviewServices.

Shared by the HTTP API and by the Streamlit UI when it runs the analysis in-process
(as on Streamlit Community Cloud, which can only run a single Streamlit process).
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta

from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from review_analyzer.cache import CachedEmbedder, CachedSentimentModel
from review_analyzer.config import Settings
from review_analyzer.ml import (
    Embedder,
    SentimentModel,
    TransformerEmbedder,
    TransformerSentiment,
    ml_available,
)
from review_analyzer.scraper import GooglePlayClient, ReviewSource
from review_analyzer.service import ReviewService, SyncPolicy
from review_analyzer.storage import (
    ModelOutputCache,
    ReviewRepository,
    SyncLog,
    init_db,
    make_engine,
)

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    engine: Engine
    source: ReviewSource
    sentiment_model: SentimentModel | None
    embedder: Embedder | None
    session_factory: sessionmaker[Session] = field(init=False)

    def __post_init__(self) -> None:
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    @property
    def policy(self) -> SyncPolicy:
        return SyncPolicy(
            ttl=timedelta(minutes=self.settings.sync_ttl_minutes),
            max_count=self.settings.max_sync_count,
            max_per_hour=self.settings.max_syncs_per_hour,
        )

    @contextmanager
    def service(self) -> Iterator[ReviewService]:
        """A ReviewService on its own database session, with models behind the output cache."""
        with self.session_factory() as session:
            cache = ModelOutputCache(session)
            yield ReviewService(
                self.source,
                ReviewRepository(session),
                SyncLog(session),
                self.policy,
                CachedSentimentModel(self.sentiment_model, cache) if self.sentiment_model else None,
                CachedEmbedder(self.embedder, cache) if self.embedder else None,
            )

    def preload(self) -> None:
        """Fetch and analyze the configured apps, so their first visitor gets instant results."""
        for app_id in self.settings.preload_apps:
            try:
                with self.service() as service:
                    service.sync(app_id, self.settings.preload_count)
                    service.analyze(app_id, self.settings.preload_count)
                logger.info("Preloaded %s", app_id)
            except Exception:
                logger.exception("Preloading %s failed", app_id)

    def start_preload(self) -> threading.Thread | None:
        if not self.settings.preload_apps:
            return None
        thread = threading.Thread(target=self.preload, name="preload", daemon=True)
        thread.start()
        return thread


def build_runtime(
    settings: Settings | None = None,
    source: ReviewSource | None = None,
    sentiment_model: SentimentModel | None = None,
    embedder: Embedder | None = None,
) -> Runtime:
    """Create the runtime and bring the database schema up to date."""
    settings = settings or Settings()
    if settings.enable_ml and ml_available():
        sentiment_model = sentiment_model or TransformerSentiment(settings.sentiment_model)
        embedder = embedder or TransformerEmbedder(settings.embedding_model)
    engine = make_engine(settings.database_url)
    init_db(engine)
    return Runtime(
        settings=settings,
        engine=engine,
        source=source or GooglePlayClient(settings.language, settings.country),
        sentiment_model=sentiment_model,
        embedder=embedder,
    )
