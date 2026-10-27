import csv
import io

from fastapi.testclient import TestClient

from tests.conftest import APP_ID, FakeSource


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_search(client: TestClient) -> None:
    response = client.get("/apps/search", params={"q": "notes"})
    assert response.status_code == 200
    assert response.json()[0]["app_id"] == APP_ID


def test_search_requires_query(client: TestClient) -> None:
    assert client.get("/apps/search", params={"q": ""}).status_code == 422


def test_analysis_before_sync_is_404(client: TestClient) -> None:
    response = client.get(f"/apps/{APP_ID}/analysis")
    assert response.status_code == 404
    assert "Sync the app first" in response.json()["detail"]


def test_sync_then_analyze(client: TestClient) -> None:
    sync = client.post(f"/apps/{APP_ID}/sync", params={"count": 10}).json()
    assert sync == {"app_id": APP_ID, "fetched": 10, "new": 10, "total_stored": 10}

    analysis = client.get(f"/apps/{APP_ID}/analysis").json()
    assert analysis["review_count"] == 10
    assert analysis["sentiment"] == {"positive": 4, "neutral": 1, "negative": 5}
    assert analysis["issues"][0]["mentions"] == 2


def test_resync_only_counts_new_reviews(client: TestClient) -> None:
    client.post(f"/apps/{APP_ID}/sync", params={"count": 5})
    sync = client.post(f"/apps/{APP_ID}/sync", params={"count": 10}).json()
    assert sync["new"] == 5
    assert sync["total_stored"] == 10


def test_analysis_limit_uses_most_recent_reviews(client: TestClient) -> None:
    client.post(f"/apps/{APP_ID}/sync", params={"count": 10})
    analysis = client.get(f"/apps/{APP_ID}/analysis", params={"limit": 2}).json()
    # r10 (rating 4) and r09 (rating 1) are the two newest.
    assert analysis["sentiment"] == {"positive": 1, "neutral": 0, "negative": 1}


def test_reviews_pagination(client: TestClient) -> None:
    client.post(f"/apps/{APP_ID}/sync", params={"count": 10})
    page = client.get(f"/apps/{APP_ID}/reviews", params={"limit": 4, "offset": 4}).json()
    assert page["total"] == 10
    assert [r["review_id"] for r in page["items"]] == ["r06", "r05", "r04", "r03"]


def test_export_csv(client: TestClient) -> None:
    client.post(f"/apps/{APP_ID}/sync", params={"count": 10})
    response = client.get(f"/apps/{APP_ID}/export.csv")

    assert response.headers["content-type"].startswith("text/csv")
    assert f"{APP_ID}_reviews.csv" in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 10
    assert rows[0]["review_id"] == "r10"


def test_rejects_invalid_app_id(client: TestClient) -> None:
    assert client.post("/apps/not-a-package/sync").status_code == 422


def test_rejects_oversized_sync(client: TestClient) -> None:
    assert client.post(f"/apps/{APP_ID}/sync", params={"count": 5000}).status_code == 422


def test_upstream_failure_is_502(client: TestClient, source: FakeSource) -> None:
    source.fail = True
    response = client.post(f"/apps/{APP_ID}/sync")
    assert response.status_code == 502
    assert "timeout" in response.json()["detail"]
