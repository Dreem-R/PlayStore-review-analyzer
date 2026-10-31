"""Drive the Streamlit UI headlessly, backed by the in-process backend and fake models."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from review_analyzer.config import Settings
from review_analyzer.runtime import build_runtime
from review_analyzer.ui import backend as backend_module
from review_analyzer.ui.backend import LocalBackend
from tests.conftest import APP_ID, FakeEmbedder, FakeSentiment, FakeSource

APP = Path(__file__).parents[1] / "src" / "review_analyzer" / "ui" / "app.py"


@pytest.fixture
def app(
    monkeypatch: pytest.MonkeyPatch, source: FakeSource, sentiment_model: FakeSentiment
) -> Iterator[AppTest]:
    monkeypatch.setenv("REVIEW_ANALYZER_PRELOAD_APPS", json.dumps([APP_ID]))
    monkeypatch.setenv("REVIEW_ANALYZER_MAX_SYNC_COUNT", "500")
    monkeypatch.delenv("REVIEW_ANALYZER_API_URL", raising=False)
    runtime = build_runtime(
        Settings(database_url="sqlite://", enable_ml=False), source, sentiment_model, FakeEmbedder()
    )
    monkeypatch.setattr(backend_module, "create_backend", lambda: LocalBackend(runtime))
    # The app's cached backend and lookups would otherwise leak between tests.
    st.cache_resource.clear()
    st.cache_data.clear()
    yield AppTest.from_file(str(APP), default_timeout=30).run()
    st.cache_resource.clear()
    st.cache_data.clear()


def test_start_page_offers_popular_apps(app: AppTest) -> None:
    assert not app.exception
    assert app.text_input(key="query").value == ""
    assert app.pills[0].options == ["Example Notes"]
    assert app.selectbox[0].options == ["100 most recent", "200 most recent", "500 most recent"]


def test_picking_a_popular_app_fills_the_search(app: AppTest) -> None:
    app.pills[0].set_value(APP_ID).run()
    assert app.text_input(key="query").value == APP_ID
    assert app.selectbox[1].options == [f"Example Notes ({APP_ID})"]


def test_search_and_analyze(app: AppTest) -> None:
    app.text_input(key="query").set_value("notes").run()
    app.button[0].click().run()

    assert not app.exception
    assert not app.error
    assert any("Fetched 10 reviews" in c.value for c in app.caption)
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Reviews analyzed"] == "10"
    assert [s.value for s in app.subheader][:3] == [
        "Sentiment",
        "Recurring complaints",
        "By app version",
    ]


def test_second_analysis_reuses_the_fetch(app: AppTest, source: FakeSource) -> None:
    app.text_input(key="query").set_value("notes").run()
    app.button[0].click().run()
    app.button[0].click().run()

    assert len(source.fetches) == 1
    assert any("Using reviews fetched" in c.value for c in app.caption)


def test_play_store_outage_shows_a_friendly_message(app: AppTest, source: FakeSource) -> None:
    app.text_input(key="query").set_value("notes").run()
    source.fail = True
    app.button[0].click().run()

    assert not app.exception
    assert "isn't answering this demo right now" in app.error[0].value
    assert "Example Notes" in app.info[0].value
