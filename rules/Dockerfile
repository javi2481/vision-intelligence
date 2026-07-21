# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY rules/requirements.txt /app/requirements-rules-sink.txt
RUN pip install --no-cache-dir -r /app/requirements-rules-sink.txt

COPY rules/ /app/rules/

EXPOSE 8850

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8850/health || exit 1

CMD ["uvicorn", "rules.app:app", "--host", "0.0.0.0", "--port", "8850", "--reload"]
