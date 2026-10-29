import csv
import io

from fastapi.testclient import TestClient

from tests.conftest import APP_ID, FakeEmbedder, FakeSentiment, FakeSource


def sync(client: TestClient, count: int = 10) -> None:
    assert client.post(f"/apps/{APP_ID}/sync", params={"count": count}).status_code == 200


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_search(client: TestClient) -> None:
    response = client.get("/apps/search", params={"q": "notes"})
    assert response.status_code == 200
    assert response.json()[0]["app_id"] == APP_ID


def test_search_requires_query(client: TestClient) -> None:
    assert client.get("/apps/search", params={"q": ""}).status_code == 422


def test_lookup(client: TestClient) -> None:
    assert client.get(f"/apps/{APP_ID}").json()["title"] == "Example Notes"


def test_lookup_unknown_app_is_404(client: TestClient) -> None:
    response = client.get("/apps/com.does.not.exist")
    assert response.status_code == 404
    assert "com.does.not.exist" in response.json()["detail"]


def test_analysis_before_sync_is_404(client: TestClient) -> None:
    response = client.get(f"/apps/{APP_ID}/analysis")
    assert response.status_code == 404
    assert "Sync the app first" in response.json()["detail"]


def test_sync_reports_new_reviews(client: TestClient) -> None:
    first = client.post(f"/apps/{APP_ID}/sync", params={"count": 5}).json()
    second = client.post(f"/apps/{APP_ID}/sync", params={"count": 10}).json()
    assert first == {"app_id": APP_ID, "fetched": 5, "new": 5, "total_stored": 5}
    assert second["new"] == 5
    assert second["total_stored"] == 10


def test_analysis_without_models_uses_ratings_and_keywords(client: TestClient) -> None:
    sync(client)
    analysis = client.get(f"/apps/{APP_ID}/analysis").json()

    assert analysis["issue_method"] == "keywords"
    assert analysis["text_sentiment"] is None
    assert analysis["rating_sentiment"] == {"positive": 4, "neutral": 1, "negative": 5}
    assert analysis["issues"][0]["label"] == "Performance"
    assert analysis["mismatches"] == []


def test_analysis_with_models(ml_client: TestClient) -> None:
    sync(ml_client)
    analysis = ml_client.get(f"/apps/{APP_ID}/analysis").json()

    assert analysis["issue_method"] == "clustering"
    assert analysis["text_sentiment"] is not None
    assert sum(analysis["text_sentiment"].values()) == 10
    # r10 is 4 stars but mentions an ad, which FakeSentiment reads as negative.
    assert [m["review_id"] for m in analysis["mismatches"]] == ["r10"]
    assert analysis["versions"][0]["version"] == "3.2.0"


def test_repeat_analysis_makes_no_model_calls(
    ml_client: TestClient, sentiment_model: FakeSentiment, embedder: FakeEmbedder
) -> None:
    sync(ml_client)
    first = ml_client.get(f"/apps/{APP_ID}/analysis").json()
    batches, embedded = len(sentiment_model.batches), embedder.calls

    second = ml_client.get(f"/apps/{APP_ID}/analysis").json()

    assert second == first
    assert len(sentiment_model.batches) == batches
    assert embedder.calls == embedded
    stored = ml_client.get(f"/apps/{APP_ID}/reviews").json()["items"]
    assert all(r["text_sentiment"] is not None for r in stored)


def test_analysis_after_sync_only_scores_new_reviews(
    ml_client: TestClient, sentiment_model: FakeSentiment
) -> None:
    sync(ml_client, count=5)
    ml_client.get(f"/apps/{APP_ID}/analysis")
    old = {r["content"] for r in ml_client.get(f"/apps/{APP_ID}/reviews").json()["items"]}
    seen = len(sentiment_model.batches)

    sync(ml_client, count=10)
    ml_client.get(f"/apps/{APP_ID}/analysis")

    new_texts = {t for batch in sentiment_model.batches[seen:] for t in batch}
    assert new_texts
    assert not new_texts & old


def test_analysis_limit_uses_most_recent_reviews(client: TestClient) -> None:
    sync(client)
    analysis = client.get(f"/apps/{APP_ID}/analysis", params={"limit": 2}).json()
    # r10 (rating 4) and r09 (rating 1) are the two newest.
    assert analysis["rating_sentiment"] == {"positive": 1, "neutral": 0, "negative": 1}


def test_reviews_pagination(client: TestClient) -> None:
    sync(client)
    page = client.get(f"/apps/{APP_ID}/reviews", params={"limit": 4, "offset": 4}).json()
    assert page["total"] == 10
    assert [r["review_id"] for r in page["items"]] == ["r06", "r05", "r04", "r03"]


def test_export_csv(client: TestClient) -> None:
    sync(client)
    response = client.get(f"/apps/{APP_ID}/export.csv")

    assert response.headers["content-type"].startswith("text/csv")
    assert f"{APP_ID}_reviews.csv" in response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == 10
    assert rows[0]["review_id"] == "r10"
    assert "text_sentiment" in rows[0]


def test_rejects_invalid_app_id(client: TestClient) -> None:
    assert client.post("/apps/not-a-package/sync").status_code == 422


def test_rejects_oversized_sync(client: TestClient) -> None:
    assert client.post(f"/apps/{APP_ID}/sync", params={"count": 5000}).status_code == 422


def test_upstream_failure_is_502(client: TestClient, source: FakeSource) -> None:
    source.fail = True
    response = client.post(f"/apps/{APP_ID}/sync")
    assert response.status_code == 502
    assert "timeout" in response.json()["detail"]
