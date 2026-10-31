import pytest

from review_analyzer.config import Settings
from review_analyzer.runtime import build_runtime
from tests.conftest import APP_ID, FakeEmbedder, FakeSentiment, FakeSource


def test_preload_fetches_and_analyzes_configured_apps(
    source: FakeSource, sentiment_model: FakeSentiment, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(
        database_url="sqlite://",
        enable_ml=False,
        preload_apps=["com.unknown.app", APP_ID],
        preload_count=5,
    )
    runtime = build_runtime(settings, source, sentiment_model, FakeEmbedder())

    thread = runtime.start_preload()
    assert thread is not None
    thread.join(timeout=10)

    # The unknown app fails (nothing to analyze) without stopping the rest.
    assert "Preloading com.unknown.app failed" in caplog.text
    assert source.fetches == [("com.unknown.app", 5), (APP_ID, 5)]
    with runtime.service() as service:
        analysis = service.analyze(APP_ID, 5)
    assert analysis.text_sentiment is not None
    # Preloading scored the reviews, so opening the app makes no model calls.
    batches = len(sentiment_model.batches)
    with runtime.service() as service:
        service.sync(APP_ID, 5)
        service.analyze(APP_ID, 5)
    assert len(sentiment_model.batches) == batches


def test_no_preload_by_default(source: FakeSource) -> None:
    runtime = build_runtime(Settings(database_url="sqlite://", enable_ml=False), source)
    assert runtime.start_preload() is None


def test_settings_build_the_sync_policy(source: FakeSource) -> None:
    settings = Settings(
        database_url="sqlite://",
        enable_ml=False,
        sync_ttl_minutes=30,
        max_sync_count=500,
        max_syncs_per_hour=7,
    )
    policy = build_runtime(settings, source).policy
    assert (policy.ttl.total_seconds(), policy.max_count, policy.max_per_hour) == (1800, 500, 7)
