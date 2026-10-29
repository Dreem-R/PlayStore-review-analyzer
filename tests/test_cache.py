from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from review_analyzer.cache import CachedEmbedder, CachedSentimentModel
from review_analyzer.storage import ModelOutputCache, init_db, make_engine
from tests.conftest import FakeEmbedder, FakeSentiment


@pytest.fixture
def engine() -> Engine:
    engine = make_engine("sqlite://")
    init_db(engine)
    return engine


@pytest.fixture
def cache(engine: Engine) -> Iterator[ModelOutputCache]:
    with Session(engine) as session:
        yield ModelOutputCache(session)


def test_sentiment_is_computed_once_per_text(cache: ModelOutputCache) -> None:
    model = FakeSentiment()
    cached = CachedSentimentModel(model, cache)

    first = cached.predict(["App crashes a lot", "Love it", "App crashes a lot"])
    second = cached.predict(["Love it", "Too slow", "App crashes a lot"])

    assert first == [("negative", 0.9), ("positive", 0.9), ("negative", 0.9)]
    assert second == [("positive", 0.9), ("negative", 0.9), ("negative", 0.9)]
    # The duplicate in the first call and the repeats in the second never reach the model.
    assert model.batches == [["App crashes a lot", "Love it"], ["Too slow"]]


def test_cache_is_per_model(cache: ModelOutputCache) -> None:
    first, second = FakeSentiment(), FakeSentiment()
    second.model_name = "another-model"

    CachedSentimentModel(first, cache).predict(["Love it"])
    CachedSentimentModel(second, cache).predict(["Love it"])

    assert second.batches == [["Love it"]]


def test_cache_survives_new_sessions(engine: Engine) -> None:
    model = FakeSentiment()
    for _ in range(2):
        with Session(engine) as session:
            CachedSentimentModel(model, ModelOutputCache(session)).predict(["Love it"])
    assert len(model.batches) == 1


def test_embeddings_round_trip(cache: ModelOutputCache) -> None:
    model = FakeEmbedder()
    cached = CachedEmbedder(model, cache)
    texts = ["Too many ads", "It crashes", "Too many ads"]

    first = cached.encode(texts)
    second = cached.encode(texts)

    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, model.encode(texts))
    assert model.calls == 2 + 3  # two unique texts, then the direct comparison call


def test_empty_input(cache: ModelOutputCache) -> None:
    assert CachedSentimentModel(FakeSentiment(), cache).predict([]) == []
    assert CachedEmbedder(FakeEmbedder(), cache).encode([]).shape == (0, 0)


def test_saving_the_same_output_twice_is_harmless(cache: ModelOutputCache) -> None:
    cache.save_sentiments("m", {"abc": ("positive", 0.9)})
    cache.save_sentiments("m", {"abc": ("positive", 0.9)})
    assert cache.sentiments("m", {"abc"}) == {"abc": ("positive", 0.9)}
