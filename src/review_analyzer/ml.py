"""Transformer models for review sentiment and sentence embeddings.

Both need the `ml` extra (torch + transformers) and load lazily on first use,
so the API starts immediately and the download happens on the first analysis.
"""

import importlib.util
import threading
from collections.abc import Iterator, Sequence
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from review_analyzer.schemas import Sentiment

SENTIMENT_LABELS: frozenset[str] = frozenset({"positive", "neutral", "negative"})


class SentimentModel(Protocol):
    model_name: str

    def predict(self, texts: Sequence[str]) -> list[tuple[Sentiment, float]]:
        """Return (label, confidence) for each text."""
        ...


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one L2-normalized row per text."""
        ...


def ml_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def _batches_by_length(texts: Sequence[str], size: int) -> Iterator[tuple[list[int], list[str]]]:
    """Yield (original indices, texts) batches of similar length.

    Each batch is padded to its longest text, so mixing a one-liner with a
    paragraph wastes most of the compute. Sorting first roughly halves inference time.
    """
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), size):
        indices = order[start : start + size]
        yield indices, [texts[i] for i in indices]


class _LazyTransformer:
    def __init__(self, model_name: str, batch_size: int) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None

    def _load_model(self) -> Any:
        raise NotImplementedError

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = self._load_model()
                model.eval()
                self._model = model

    def _tokenize(self, batch: list[str]) -> Any:
        return self._tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        )


class TransformerSentiment(_LazyTransformer):
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        super().__init__(model_name, batch_size)
        self._labels: dict[int, Sentiment] = {}

    def _load_model(self) -> Any:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        labels = {int(i): str(name).lower() for i, name in model.config.id2label.items()}
        if set(labels.values()) != SENTIMENT_LABELS:
            raise ValueError(f"{self.model_name} must predict {sorted(SENTIMENT_LABELS)}")
        self._labels = cast(dict[int, Sentiment], labels)
        return model

    def predict(self, texts: Sequence[str]) -> list[tuple[Sentiment, float]]:
        import torch

        self._ensure_loaded()
        predictions: list[tuple[Sentiment, float]] = [("neutral", 0.0)] * len(texts)
        for indices, batch in _batches_by_length(texts, self.batch_size):
            with torch.no_grad():
                probs = self._model(**self._tokenize(batch)).logits.softmax(dim=-1)
            scores, ids = probs.max(dim=-1)
            for index, label_id, score in zip(indices, ids.tolist(), scores.tolist(), strict=True):
                predictions[index] = (self._labels[label_id], round(score, 4))
        return predictions


class TransformerEmbedder(_LazyTransformer):
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        super().__init__(model_name, batch_size)

    def _load_model(self) -> Any:
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.model_name)

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        import torch

        self._ensure_loaded()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: NDArray[np.float32] | None = None
        for indices, batch in _batches_by_length(texts, self.batch_size):
            inputs = self._tokenize(batch)
            with torch.no_grad():
                hidden = self._model(**inputs).last_hidden_state
            # Mean-pool over real tokens (not padding), then normalize for cosine similarity.
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1).numpy().astype(np.float32)
            if vectors is None:
                vectors = np.zeros((len(texts), pooled.shape[1]), dtype=np.float32)
            vectors[indices] = pooled
        assert vectors is not None
        return vectors
