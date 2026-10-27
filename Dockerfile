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

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --extra ml --extra ui

RUN useradd --create-home app && mkdir -p data .cache && chown -R app data .cache
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "review_analyzer.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
