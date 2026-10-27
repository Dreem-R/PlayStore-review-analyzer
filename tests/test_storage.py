from review_analyzer.schemas import Review
from review_analyzer.storage import ReviewRepository


def test_upsert_reports_only_new_reviews(
    repository: ReviewRepository, reviews: list[Review]
) -> None:
    assert repository.upsert(reviews[:6]) == 6
    assert repository.upsert(reviews) == 4
    assert repository.count("com.example.notes") == 10


def test_upsert_updates_edited_reviews(repository: ReviewRepository, reviews: list[Review]) -> None:
    repository.upsert(reviews)
    edited = reviews[1].model_copy(update={"rating": 4, "content": "Fixed in 3.2.1, thanks!"})

    assert repository.upsert([edited]) == 0
    stored = {r.review_id: r for r in repository.latest("com.example.notes", 100)}
    assert stored[edited.review_id].rating == 4
    assert stored[edited.review_id].content == "Fixed in 3.2.1, thanks!"


def test_upsert_deduplicates_within_a_batch(
    repository: ReviewRepository, reviews: list[Review]
) -> None:
    assert repository.upsert([reviews[0], reviews[0]]) == 1


def test_latest_is_newest_first_and_paginated(
    repository: ReviewRepository, reviews: list[Review]
) -> None:
    repository.upsert(reviews)

    first_page = repository.latest("com.example.notes", limit=3)
    second_page = repository.latest("com.example.notes", limit=3, offset=3)

    assert [r.review_id for r in first_page] == ["r10", "r09", "r08"]
    assert [r.review_id for r in second_page] == ["r07", "r06", "r05"]


def test_reviews_are_scoped_by_app(repository: ReviewRepository, reviews: list[Review]) -> None:
    other = [
        r.model_copy(update={"review_id": f"x-{r.review_id}", "app_id": "com.other.app"})
        for r in reviews[:3]
    ]
    repository.upsert(reviews + other)

    assert repository.count("com.example.notes") == 10
    assert repository.count("com.other.app") == 3
    assert {r.app_id for r in repository.latest("com.other.app", 100)} == {"com.other.app"}
