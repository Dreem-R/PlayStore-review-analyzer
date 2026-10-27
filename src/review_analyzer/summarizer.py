"""Optional abstractive summarizer. Needs the `ml` extra (transformers + torch)."""

import importlib.util
import threading
from typing import Any


class LazySummarizer:
    """Loads the model on first use, so the API starts without waiting for a download."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("transformers") is not None

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

    def __call__(self, text: str) -> str:
        if self._model is None:
            self._load()
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        output = self._model.generate(**inputs, max_length=100, min_length=30, num_beams=4)
        return str(self._tokenizer.decode(output[0], skip_special_tokens=True)).strip()
