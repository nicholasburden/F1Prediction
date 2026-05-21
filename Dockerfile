# syntax=docker/dockerfile:1.7

# ---------- Build stage: install deps + project into a venv ----------
FROM python:3.13-slim AS builder

# Grab the uv binary from its official image — fastest install + reproducible.
COPY --from=ghcr.io/astral-sh/uv:0.4.18 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

# Install third-party deps first so source-only changes don't bust the cache.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Now bring in the project and install it on top.
COPY f1prediction ./f1prediction
COPY scripts ./scripts
COPY webapp ./webapp
COPY main.py README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---------- Runtime stage: minimal image with the venv + source ----------
FROM python:3.13-slim AS runtime

# libgomp is pulled in by torch and friends; install ahead of copy so caching
# is stable across source-only rebuilds.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
