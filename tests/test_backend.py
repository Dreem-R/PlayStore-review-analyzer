import json

import httpx
import pytest
from fastapi.testclient import TestClient

from review_analyzer.config import Settings
from review_analyzer.runtime import build_runtime
from review_analyzer.ui.backend import Backend, BackendError, HttpBackend, LocalBackend
from tests.conftest import APP_ID, FakeEmbedder, FakeSentiment, FakeSource


@pytest.fixture
def local(source: FakeSource, sentiment_model: FakeSentiment) -> LocalBackend:
    settings = Settings(database_url="sqlite://", enable_ml=False)
    return LocalBackend(build_runtime(settings, source, sentiment_model, FakeEmbedder()))


@pytest.fixture
def http(ml_client: TestClient) -> HttpBackend:
    # TestClient is an httpx.Client wired to the FastAPI app: this exercises the real API.
    return HttpBackend("http://testserver", client=ml_client)


@pytest.fixture(params=["local", "http"])
def backend(request: pytest.FixtureRequest) -> Backend:
    backend: Backend = request.getfixturevalue(request.param)
    return backend


def test_full_flow_returns_json_ready_data(backend: Backend) -> None:
    assert backend.search("notes")[0]["app_id"] == APP_ID
    assert backend.lookup(APP_ID)["title"] == "Example Notes"

    sync = backend.sync(APP_ID, 10)
    analysis = backend.analysis(APP_ID, 10)
    reviews = backend.reviews(APP_ID, 10)
    csv = backend.export_csv(APP_ID, 10)

    assert sync["status"] == "fetched"
    assert analysis["review_count"] == 10
    assert analysis["issue_method"] == "clustering"
    assert len(reviews) == 10
    assert csv.startswith("review_id,")
    # Both backends hand the UI plain JSON types, e.g. ISO strings for datetimes.
    json.dumps([sync, analysis, reviews])
    assert isinstance(sync["last_synced_at"], str)


def test_unknown_app_is_not_found(backend: Backend) -> None:
    with pytest.raises(BackendError) as error:
        backend.lookup("com.does.not.exist")
    assert error.value.kind == "not_found"


def test_analysis_before_sync_is_not_found(backend: Backend) -> None:
    with pytest.raises(BackendError) as error:
        backend.analysis(APP_ID, 10)
    assert error.value.kind == "not_found"


def test_play_store_failure_is_unavailable(backend: Backend, source: FakeSource) -> None:
    source.fail = True
    with pytest.raises(BackendError) as error:
        backend.sync(APP_ID, 10)
    assert error.value.kind == "unavailable"


@pytest.mark.parametrize(
    ("status", "kind"),
    [(429, "unavailable"), (502, "unavailable"), (404, "not_found"), (500, "other")],
)
def test_http_status_maps_to_error_kind(status: int, kind: str) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status, json={"detail": "x"}))
    client = httpx.Client(base_url="http://api", transport=transport)
    with pytest.raises(BackendError) as error:
        HttpBackend("http://api", client=client).lookup(APP_ID)
    assert error.value.kind == kind


def test_unreachable_api_is_reported() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    client = httpx.Client(base_url="http://api", transport=httpx.MockTransport(refuse))
    with pytest.raises(BackendError, match="Can't reach the API"):
        HttpBackend("http://api", client=client).search("notes")
