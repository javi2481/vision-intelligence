"""
Adaptador FastAPI — única pieza de código propio (Vision Intelligence Sprint 1).

Responsabilidad: normalizar detecciones PaddleX → PerceptionEvent (epp-core).
NO decide reglas de negocio (JetLinks) ni renderiza UI (AMIS).

Flujo:
  POST /ingest  → acumula por track_id en track_cache (TTL)
  sweeper TTL   → consolidate_and_emit → events_buffer
  GET  /events  → buffer para AMIS/ECharts
  POST /webhook/rules → gancho listo para JetLinks
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from epp_core import PerceptionEvent

logger = logging.getLogger("adapter")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# --- Configuración vía entorno (portable edge / docker) ---
TRACK_TTL_SECONDS = float(os.getenv("TRACK_TTL_SECONDS", "10"))
MAX_DETECTIONS_PER_TRACK = int(os.getenv("MAX_DETECTIONS_PER_TRACK", "60"))
EVENTS_BUFFER_SIZE = int(os.getenv("EVENTS_BUFFER_SIZE", "500"))
SWEEP_INTERVAL_SECONDS = float(os.getenv("SWEEP_INTERVAL_SECONDS", "1.0"))
JETLINKS_WEBHOOK_URL = os.getenv("JETLINKS_WEBHOOK_URL", "")  # vacío = modo MVP local
JETLINKS_API_KEY = os.getenv("JETLINKS_API_KEY", "demo")
STATIC_DIR = os.getenv("STATIC_DIR", os.path.dirname(os.path.abspath(__file__)))


class TrackBucket(BaseModel):
    """Caché en memoria de un track: detecciones + metadatos de TTL."""

    detections: list[dict[str, Any]] = Field(default_factory=list)
    last_seen: float = Field(default_factory=time.monotonic)
    finalized: bool = False


# Estado en proceso (delgado; mañana en edge = mismo dict)
track_cache: dict[str, TrackBucket] = {}
events_buffer: deque[dict[str, Any]] = deque(maxlen=EVENTS_BUFFER_SIZE)
_stats = {"ingested": 0, "emitted": 0, "paddlex_degraded": False}


class IngestBody(BaseModel):
    """Cuerpo flexible: lista de detecciones o envelope PaddleX."""

    detections: Optional[list[dict[str, Any]]] = None
    result: Optional[list[dict[str, Any]]] = None
    data: Optional[list[dict[str, Any]]] = None
    # Campos sueltos: permitir POST de una sola detección
    track_id: Optional[str] = None

    model_config = {"extra": "allow"}


def _extract_detections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrae lista de detecciones de envelopes comunes de PaddleX."""
    for key in ("detections", "result", "data", "vehicles", "boxes"):
        value = payload.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    # Una sola detección en raíz
    if payload.get("track_id") is not None or payload.get("bbox") is not None:
        return [payload]
    return []


def _append_to_track(track_id: str, detection: dict[str, Any]) -> None:
    """Agrega detección al caché del track; respeta soft-cap y refresca TTL."""
    bucket = track_cache.get(track_id)
    if bucket is None:
        bucket = TrackBucket()
        track_cache[track_id] = bucket

    if bucket.finalized:
        return

    if len(bucket.detections) >= MAX_DETECTIONS_PER_TRACK:
        # Mantener las más recientes (ventana deslizante)
        bucket.detections = bucket.detections[-(MAX_DETECTIONS_PER_TRACK - 1) :]

    detection = {**detection, "track_id": track_id}
    bucket.detections.append(detection)
    bucket.last_seen = time.monotonic()


def _emit_track(track_id: str) -> list[PerceptionEvent]:
    """Consolida un track y lo saca del caché."""
    bucket = track_cache.pop(track_id, None)
    if bucket is None or not bucket.detections:
        return []

    events = PerceptionEvent.consolidate_and_emit(bucket.detections)
    for event in events:
        events_buffer.appendleft(event.model_dump(mode="json"))
        _stats["emitted"] += 1
        logger.info(
            "Emitted PerceptionEvent track=%s candidates=%s conf=%.3f",
            track_id,
            event.candidate_ids,
            event.confidence,
        )
    return events


async def _forward_to_jetlinks(events: list[PerceptionEvent]) -> None:
    """
    Gancho JetLinks: si JETLINKS_WEBHOOK_URL está definido, POST HTTP.
    En MVP sin JetLinks, solo loguea (contrato idéntico).
    """
    if not events:
        return

    payload = [e.model_dump(mode="json") for e in events]

    if not JETLINKS_WEBHOOK_URL:
        for event in events:
            logger.info(
                "[rules-mvp] candidate_ids=%s conf=%.3f plate=%s",
                event.candidate_ids,
                event.confidence,
                event.payload.plate_text,
            )
        return

    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                JETLINKS_WEBHOOK_URL,
                json=payload,
                headers={"x-api-key": JETLINKS_API_KEY},
            )
            resp.raise_for_status()
            logger.info("Forwarded %d events to JetLinks", len(events))
    except Exception as exc:  # degradación elegante
        logger.warning("JetLinks unreachable (%s); events kept in buffer", exc)


async def _sweep_expired_tracks() -> None:
    """Emite tracks cuyo TTL expiró o están marcados finalizados."""
    now = time.monotonic()
    expired = [
        tid
        for tid, bucket in track_cache.items()
        if bucket.finalized or (now - bucket.last_seen) >= TRACK_TTL_SECONDS
    ]
    for tid in expired:
        events = _emit_track(tid)
        await _forward_to_jetlinks(events)


async def _sweeper_loop(stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await _sweep_expired_tracks()
        except Exception:
            logger.exception("Sweeper error")
        try:
            await asyncio.wait_for(stop.wait(), timeout=SWEEP_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    stop = asyncio.Event()
    task = asyncio.create_task(_sweeper_loop(stop))
    logger.info(
        "Adapter started TTL=%.1fs buffer=%d",
        TRACK_TTL_SECONDS,
        EVENTS_BUFFER_SIZE,
    )
    yield
    stop.set()
    await task


app = FastAPI(
    title="Vision Intelligence Adapter",
    description="Normaliza PaddleX → epp-core (schema 1.0-draft)",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Healthcheck para Docker Compose."""
    return {
        "status": "ok",
        "tracks_active": len(track_cache),
        "events_buffered": len(events_buffer),
        "paddlex_degraded": _stats["paddlex_degraded"],
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    """
    Recibe detecciones JSON de PaddleX (vía bridge) y las agrega al track_cache.
    No emite PerceptionEvent aquí: el sweeper TTL consolida al expirar el track.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)

    if not isinstance(body, dict):
        if isinstance(body, list):
            body = {"detections": body}
        else:
            return JSONResponse({"ok": False, "error": "expected object"}, status_code=400)

    # Señal de degradación desde el bridge
    if body.get("degraded") is True:
        _stats["paddlex_degraded"] = True
        return JSONResponse({"ok": True, "degraded": True, "accepted": 0})

    detections = _extract_detections(body)
    if not detections:
        return JSONResponse({"ok": True, "accepted": 0, "message": "no detections"})

    _stats["paddlex_degraded"] = False
    accepted = 0
    for det in detections:
        track_id = det.get("track_id") or det.get("tracker_id")
        if track_id is None:
            continue
        _append_to_track(str(track_id), det)
        accepted += 1
        _stats["ingested"] += 1

        # Finalización explícita (PaddleX / bridge puede marcar lost track)
        if det.get("finalized") or det.get("track_lost"):
            track_cache[str(track_id)].finalized = True

    # Sweep inmediato de finalizados
    await _sweep_expired_tracks()

    return JSONResponse(
        {
            "ok": True,
            "accepted": accepted,
            "tracks_active": len(track_cache),
        }
    )


@app.get("/events")
async def get_events(limit: int = 100) -> dict[str, Any]:
    """Buffer de PerceptionEvent para AMIS / ECharts."""
    limit = max(1, min(limit, EVENTS_BUFFER_SIZE))
    items = list(events_buffer)[:limit]
    return {
        "count": len(items),
        "total_emitted": _stats["emitted"],
        "tracks_active": len(track_cache),
        "degraded": _stats["paddlex_degraded"],
        "events": items,
    }


@app.post("/webhook/rules")
async def webhook_rules(request: Request) -> JSONResponse:
    """
    Preparación JetLinks: acepta PerceptionEvent(s) y aplica/forward reglas.
    Contrato estable — cambiar solo JETLINKS_WEBHOOK_URL en producción.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)

    raw_events = body if isinstance(body, list) else body.get("events", [body])
    parsed: list[PerceptionEvent] = []
    for item in raw_events:
        try:
            parsed.append(PerceptionEvent.model_validate(item))
        except Exception as exc:
            logger.warning("Invalid PerceptionEvent skipped: %s", exc)

    await _forward_to_jetlinks(parsed)
    return JSONResponse({"ok": True, "processed": len(parsed)})


@app.get("/")
async def dashboard() -> FileResponse:
    """Sirve el shell HTML que carga AMIS + amis_dashboard.json."""
    path = os.path.join(STATIC_DIR, "dashboard.html")
    return FileResponse(path)


# Assets estáticos (amis_dashboard.json, etc.)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
