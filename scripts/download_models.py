"""Download the sentiment and embedding models into the Hugging Face cache.

Used by the Dockerfile so the image ships with the models and the first
analysis doesn't wait on a ~600 MB download. Kept free of project imports so
that Docker layer can be cached independently of source changes.

Usage: python scripts/download_models.py SENTIMENT_MODEL EMBEDDING_MODEL
"""

import sys

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def main(sentiment_model: str, embedding_model: str) -> None:
    AutoTokenizer.from_pretrained(sentiment_model)
    AutoModelForSequenceClassification.from_pretrained(sentiment_model)
    AutoTokenizer.from_pretrained(embedding_model)
    AutoModel.from_pretrained(embedding_model)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
