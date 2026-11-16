ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG TARGETARCH

FROM --platform=$BUILDPLATFORM node:22.23.4-alpine AS web-build

WORKDIR /app/web

COPY web/package.json web/bun.lock ./
# Keep downloads across a failed BuildKit step and bound npm transport retries.
# Registry, TLS, and proxy selection remain unchanged.
RUN --mount=type=cache,target=/root/.npm,sharing=locked \
    npm install --maxsockets=5 --fetch-retries=5 --fetch-retry-factor=2 \
      --fetch-retry-mintimeout=1000 --fetch-retry-maxtimeout=30000

COPY VERSION /app/VERSION
COPY CHANGELOG.md /app/CHANGELOG.md
COPY web ./
RUN NEXT_PUBLIC_APP_VERSION="$(cat /app/VERSION)" npm run build


FROM --platform=$TARGETPLATFORM python:3.13-slim AS app

ARG TARGETPLATFORM
ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 安装系统依赖
# - git: Git 存储后端需要
# - libpq-dev: PostgreSQL 客户端库
# - gcc: 编译 psycopg2-binary 需要
# Retry transient repository/proxy failures without changing sources or trust.
# Timeouts bound each connection/data wait; failed updates still stop the build.
RUN apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
      -o Acquire::https::Timeout=30 -o APT::Update::Error-Mode=any update \
    && apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 \
      -o Acquire::https::Timeout=30 install -y --no-install-recommends \
    ca-certificates \
    git \
    libpq-dev \
    gcc \
    openssl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade 'pip==26.2.2' 'uv==0.13.4'

COPY pyproject.toml uv.lock ./
# Keep dependency downloads bounded on the shared build proxy.
# Preserve locked URLs, versions and hashes; final failures stop the build.
RUN UV_CONCURRENT_DOWNLOADS=4 UV_HTTP_CONNECT_TIMEOUT=15 \
    UV_HTTP_TIMEOUT=60 UV_HTTP_RETRIES=3 \
    uv sync --frozen --no-dev --no-install-project

COPY main.py ./
COPY config.json ./
COPY VERSION ./
COPY api ./api
COPY services ./services
COPY utils ./utils
COPY scripts ./scripts
COPY --from=web-build /app/web/out ./web_dist

EXPOSE 80

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--access-log"]
