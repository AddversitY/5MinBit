# Paper-testing engine image. Read-only market data; no keys required.
FROM node:22-bookworm-slim AS web
WORKDIR /app/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
COPY scripts ./scripts
RUN pip install .
COPY --from=web /app/dashboard/dist ./dashboard/dist
# data/ (raw captures, ledger, models, reports) must be a persistent volume
VOLUME ["/app/data"]
EXPOSE 8000
HEALTHCHECK --interval=60s --timeout=10s CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"
CMD ["btc5m", "serve", "--host", "0.0.0.0", "--port", "8000"]
