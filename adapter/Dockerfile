# syntax=docker/dockerfile:1

# --- Stage 1: build de la SPA (Fase 1, addendum-s2-spa-s3) ---
# Node solo existe en este stage — el runtime Python final no lo necesita
# (DoD: `docker compose up --build` sirve la SPA sin Node en el host).
FROM node:20-slim AS spa-build
WORKDIR /spa-src

COPY adapter/ui/spa-src/package.json adapter/ui/spa-src/package-lock.json ./
RUN npm ci

COPY adapter/ui/spa-src/ ./
# contracts/epp.gen.ts es la fuente de tipos (generado por
# scripts/gen_epp_types.py desde adapter/epp_core.py, ver CI). Se copia
# directo en vez de correr scripts/copy-types.mjs (ese script asume el
# layout del repo en el host, no el WORKDIR de este stage).
COPY contracts/epp.gen.ts ./src/types/epp.gen.ts
RUN npx tsc -b && npx vite build
# vite.config.ts: build.outDir = "../spa" (relativo a /spa-src) -> /spa

# --- Stage 2: runtime Python (FastAPI adapter) ---
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    STATIC_DIR=/app/adapter/ui \
    SPA_DIR=/app/adapter/ui/spa

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY adapter/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY adapter/ /app/adapter/
# Pisa el placeholder .gitkeep del repo con el build real de la SPA.
COPY --from=spa-build /spa /app/adapter/ui/spa

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "adapter.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
