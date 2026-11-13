FROM --platform=$BUILDPLATFORM node:22-alpine AS web
WORKDIR /web
COPY web/package*.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12.13-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-default-groups --extra docker-cpu --no-install-project
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-default-groups --extra docker-cpu --no-editable
COPY --from=web /web/dist ./web/dist

# A fresh named volume inherits this mode; Linux launchers use the checkout owner's UID.
RUN useradd --create-home --uid 1000 ragglet && install -d -m 1777 /home/ragglet/.cache && chmod 755 /home/ragglet
# torch uses getpass.getuser() to locate its compile cache also for numeric UIDs.
ENV HOME=/home/ragglet USER=ragglet PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 RAGGLET_UI_DIR=/app/web/dist
ARG RAGGLET_COMMIT
ENV RAGGLET_COMMIT=${RAGGLET_COMMIT}
USER ragglet
CMD ["python", "-m", "ragglet.server"]
