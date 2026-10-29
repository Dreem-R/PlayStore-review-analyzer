"""Model wrappers that memoize outputs in the database.

Sentence-level sentiment is over 90% of an analysis's run time. With these wrappers
a repeat analysis makes no model calls, and analyzing after a sync only pays
for the new reviews. Entries are keyed by model name and a hash of the text, so
an edited review (new text) or a different model naturally misses the cache.
"""

import hashlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from review_analyzer.ml import Embedder, SentimentModel
from review_analyzer.schemas import Sentiment
from review_analyzer.storage import ModelOutputCache


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedSentimentModel:
    def __init__(self, model: SentimentModel, cache: ModelOutputCache) -> None:
        self.model_name = model.model_name
        self._model = model
        self._cache = cache

    def predict(self, texts: Sequence[str]) -> list[tuple[Sentiment, float]]:
        hashes = [text_hash(t) for t in texts]
        known = self._cache.sentiments(self.model_name, set(hashes))
        # Dict keyed by hash, so a sentence repeated across reviews is computed once.
        missing = {h: t for h, t in zip(hashes, texts, strict=True) if h not in known}
        if missing:
            computed = dict(zip(missing, self._model.predict(list(missing.values())), strict=True))
            self._cache.save_sentiments(self.model_name, computed)
            known |= computed
        return [known[h] for h in hashes]


class CachedEmbedder:
    def __init__(self, model: Embedder, cache: ModelOutputCache) -> None:
        self.model_name = model.model_name
        self._model = model
        self._cache = cache

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        hashes = [text_hash(t) for t in texts]
        known = self._cache.embeddings(self.model_name, set(hashes))
        missing = {h: t for h, t in zip(hashes, texts, strict=True) if h not in known}
        if missing:
            vectors = self._model.encode(list(missing.values()))
            computed = dict(zip(missing, vectors, strict=True))
            self._cache.save_embeddings(self.model_name, computed)
            known |= computed
        return np.vstack([known[h] for h in hashes])
