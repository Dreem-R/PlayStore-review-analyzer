FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.12 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Dependencies first so code changes don't invalidate the (large) torch layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --extra ml --extra ui

RUN useradd --create-home app && mkdir -p data .cache && chown app data .cache

# Bake the models into the image so the first analysis doesn't download ~600 MB.
# Downloaded as the app user: chown-ing them afterwards would copy them into a second layer.
# Must match the defaults in config.py.
ARG SENTIMENT_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COPY scripts/download_models.py ./scripts/
USER app
RUN python scripts/download_models.py "$SENTIMENT_MODEL" "$EMBEDDING_MODEL"
USER root

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra ml --extra ui

USER app

# The models are already here; don't reach out to the Hub at runtime.
ENV HF_HUB_OFFLINE=1 \
    REVIEW_ANALYZER_SENTIMENT_MODEL=$SENTIMENT_MODEL \
    REVIEW_ANALYZER_EMBEDDING_MODEL=$EMBEDDING_MODEL

EXPOSE 8000 8501
CMD ["uvicorn", "review_analyzer.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
