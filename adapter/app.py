"""
Adaptador FastAPI — media, ingest, consolidación y UI estática.

Responsabilidad: normalizar detecciones (vehicles/objects/plates vía bridge)
→ PerceptionEvent (epp_core). Sirve el panel AMIS desde adapter/ui/.

NO decide reglas de negocio (JetLinks/rules) ni corre inferencia PaddleX.

Flujo:
  POST /ingest  → acumula por track_id en track_cache (TTL)
  sweeper TTL   → consolidate_and_emit → events_buffer
  GET  /events  → buffer para AMIS/ECharts
  POST /webhook/rules → gancho listo para JetLinks
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from adapter.epp_core import PerceptionEvent

# ContextVar para correlacionar logs de ingest/sweep con un trace_id corto.
_trace_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "trace_id", default=None
)


class _TraceIdFilter(logging.Filter):
    """Inyecta record.trace_id para el format string (vacío si no hay)."""

    def filter(self, record: logging.LogRecord) -> bool:
        tid = _trace_id_var.get()
        record.trace_id = f"trace_id={tid} " if tid is not None else ""  # type: ignore[attr-defined]
        return True


logger = logging.getLogger("adapter")
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(trace_id)s%(message)s")
)
_handler.addFilter(_TraceIdFilter())
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[_handler], force=True)

# --- Configuración vía entorno (portable edge / docker) ---
TRACK_TTL_SECONDS = float(os.getenv("TRACK_TTL_SECONDS", "10"))
MAX_DETECTIONS_PER_TRACK = int(os.getenv("MAX_DETECTIONS_PER_TRACK", "60"))
EVENTS_BUFFER_SIZE = int(os.getenv("EVENTS_BUFFER_SIZE", "500"))
SWEEP_INTERVAL_SECONDS = float(os.getenv("SWEEP_INTERVAL_SECONDS", "1.0"))
JETLINKS_WEBHOOK_URL = os.getenv("JETLINKS_WEBHOOK_URL", "")  # vacío = modo MVP local
JETLINKS_API_KEY = os.getenv("JETLINKS_API_KEY", "demo")
# UI estática vive en adapter/ui/ (dashboard, AMIS schema, placeholder).
_DEFAULT_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
STATIC_DIR = os.getenv("STATIC_DIR", _DEFAULT_STATIC)
RULES_SINK_URL = os.getenv("RULES_SINK_URL", "http://rules-sink:8850")
VI_ENV = os.getenv("VI_ENV", "development").strip().lower()


def _enforce_production_secrets() -> None:
    """En VI_ENV=production exige JETLINKS_API_KEY fuerte (!= demo / vacío)."""
    if VI_ENV != "production":
        return
    key = (JETLINKS_API_KEY or "").strip()
    if not key or key == "demo":
        raise SystemExit(
            "VI_ENV=production requires JETLINKS_API_KEY set to a non-demo secret"
        )


_enforce_production_secrets()

# --- Selector de muestra local (solo fotos) + preview overlay EN ---
# MEDIA_DIR/images <- ./imagenes_muestra (montado RO en docker-compose).
# Solo JPGs (etc.) en la raíz de images; no recursivo (no indexa DETRAC).
# El allow-list es SIEMPRE lo que existe físicamente ahí; nunca se acepta
# una ruta arbitraria del cliente (ver `_find_media_path`).
# Watcher: al aparecer/actualizar un JPG, auto-selecciona el más nuevo (mtime).
MEDIA_DIR = os.getenv("MEDIA_DIR", "/media")
MEDIA_IMAGE_SUBDIR = "images"
MEDIA_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
MEDIA_WATCH_INTERVAL = float(os.getenv("MEDIA_WATCH_INTERVAL", "1.0"))
PREVIEW_MJPEG_INTERVAL = float(os.getenv("PREVIEW_MJPEG_INTERVAL", "0.05"))


class TrackBucket(BaseModel):
    """Caché en memoria de un track: detecciones + metadatos de TTL."""

    detections: list[dict[str, Any]] = Field(default_factory=list)
    last_seen: float = Field(default_factory=time.monotonic)
    finalized: bool = False


@dataclass
class AppState:
    """Estado mutable del proceso, inyectado en lifespan."""

    track_cache: dict[str, TrackBucket] = field(default_factory=dict)
    events_buffer: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=EVENTS_BUFFER_SIZE)
    )
    stats: dict[str, Any] = field(
        default_factory=lambda: {
            "ingested": 0,
            "emitted": 0,
            "paddlex_degraded": False,
        }
    )
    current_media: Optional[dict[str, str]] = None
    generation: int = 0
    # Última generación efectivamente ingerida por el bridge (via trace_id/
    # generation en /ingest). None = todavía sin ningún ingest. Completitud
    # (para la SPA) es DERIVADA: generation == last_ingest_generation.
    # No se resetea en `_flush_detection_session` / bump de `generation`:
    # queda "stale" (de la generación anterior) hasta el próximo /ingest que
    # confirme la nueva — así el cliente puede distinguir "processing"
    # (generation != last_ingest_generation) de "completo".
    last_ingest_generation: Optional[int] = None
    latest_frame: Optional[bytes] = None
    frame_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    media_seen_mtimes: dict[str, float] = field(default_factory=dict)
    media_watch_bootstrapped: bool = False


_app_state: Optional[AppState] = None


def state() -> AppState:
    """Devuelve el AppState activo; falla si lifespan aún no arrancó."""
    if _app_state is None:
        raise RuntimeError("AppState not started")
    return _app_state


def _load_placeholder_jpeg() -> bytes:
    """JPEG de marca para preview vacío / tras Limpiar foto."""
    path = os.path.join(STATIC_DIR, "placeholder_preview.jpg")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        if data:
            return data
    except OSError as exc:
        logger.warning("Placeholder preview missing at %s (%s)", path, exc)
    return b""


_PLACEHOLDER_JPEG = _load_placeholder_jpeg()


def _list_media_items() -> list[dict[str, str]]:
    """Enumera imágenes reales (no recursivo) en {MEDIA_DIR}/images.

    Este es EL allow-list: nunca se acepta un nombre que no aparezca acá.
    No recorre subcarpetas (evita enumerar DETRAC u otros datasets anidados).
    """
    items: list[dict[str, str]] = []
    folder = os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR)
    if not os.path.isdir(folder):
        return items
    for entry in sorted(os.listdir(folder)):
        full_path = os.path.join(folder, entry)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(entry)[1].lower() not in MEDIA_IMAGE_EXTENSIONS:
            continue
        items.append({"name": entry, "type": "image"})
    return items


def _scan_media_mtimes() -> dict[str, float]:
    """Basename → mtime de imágenes en la raíz de MEDIA_DIR/images."""
    out: dict[str, float] = {}
    folder = os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR)
    if not os.path.isdir(folder):
        return out
    for entry in os.listdir(folder):
        full_path = os.path.join(folder, entry)
        if not os.path.isfile(full_path):
            continue
        if os.path.splitext(entry)[1].lower() not in MEDIA_IMAGE_EXTENSIONS:
            continue
        try:
            out[entry] = os.path.getmtime(full_path)
        except OSError:
            continue
    return out


def _pick_newest_name(mtimes: dict[str, float]) -> Optional[str]:
    """Nombre con mayor mtime; None si vacío. Empate: orden lexicográfico."""
    if not mtimes:
        return None
    return max(mtimes.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _media_changes_detected(
    seen: dict[str, float], current: dict[str, float]
) -> bool:
    """True si hay archivo nuevo o mtime mayor que el último visto."""
    for name, mtime in current.items():
        prev = seen.get(name)
        if prev is None or mtime > prev:
            return True
    return False


def _flush_detection_session() -> None:
    """Limpia buffer/tracks y deja el preview en el placeholder de marca.

    Deliberadamente NO toca `st.last_ingest_generation` (queda stale, de la
    generación anterior) — así `generation != last_ingest_generation` marca
    "processing" hasta que un nuevo /ingest confirme la generación activa.
    """
    st = state()
    st.events_buffer.clear()
    st.track_cache.clear()
    st.stats["ingested"] = 0
    st.stats["emitted"] = 0
    st.stats["paddlex_degraded"] = False
    # Empuja marca al MJPEG: si queda None el browser conserva el último frame.
    st.latest_frame = _PLACEHOLDER_JPEG or None


def _apply_media_selection(name: str) -> Optional[dict[str, Any]]:
    """Selecciona foto allow-listed; bump generation. None si inválida."""
    st = state()
    resolved = _find_media_path(name)
    if resolved is None:
        return None
    path, media_type = resolved
    if media_type != "image":
        return None
    _flush_detection_session()
    st.current_media = {"name": name, "type": "image"}
    st.generation += 1
    logger.info(
        "Media selected: %s (type=image, gen=%d, path=%s) trace_id=%s",
        name,
        st.generation,
        path,
        st.generation,
    )
    return {"ok": True, "name": name, "generation": st.generation}


def _find_media_path(name: str) -> Optional[tuple[str, str]]:
    """Resuelve `name` contra el allow-list de imágenes. None si no matchea.

    Bloquea path traversal: solo se acepta un basename tal cual aparece en
    `_list_media_items()` (mismo nombre físico presente en MEDIA_DIR/images).
    """
    if not name or os.path.basename(name) != name:
        return None
    for item in _list_media_items():
        if item["name"] == name:
            return os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR, name), "image"
    return None


class MediaSelectBody(BaseModel):
    name: str


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
    cache = state().track_cache
    bucket = cache.get(track_id)
    if bucket is None:
        bucket = TrackBucket()
        cache[track_id] = bucket

    if bucket.finalized:
        return

    if len(bucket.detections) >= MAX_DETECTIONS_PER_TRACK:
        # Mantener las más recientes (ventana deslizante)
        bucket.detections = bucket.detections[-(MAX_DETECTIONS_PER_TRACK - 1) :]

    detection = {**detection, "track_id": track_id}
    bucket.detections.append(detection)
    bucket.last_seen = time.monotonic()


def _flush_track(track_id: str) -> list[PerceptionEvent]:
    """Saca un track del caché y pide a epp_core que construya PerceptionEvent(s).

    No confundir con `epp_core._emit_track`, que arma el payload por entity_type.
    """
    st = state()
    bucket = st.track_cache.pop(track_id, None)
    if bucket is None or not bucket.detections:
        return []

    events = PerceptionEvent.consolidate_and_emit(bucket.detections)
    for event in events:
        st.events_buffer.appendleft(event.model_dump(mode="json"))
        st.stats["emitted"] += 1
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
                getattr(event.payload, "plate_text", None),
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
    cache = state().track_cache
    expired = [
        tid
        for tid, bucket in cache.items()
        if bucket.finalized or (now - bucket.last_seen) >= TRACK_TTL_SECONDS
    ]
    for tid in expired:
        events = _flush_track(tid)
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


async def _media_watch_loop(stop: asyncio.Event) -> None:
    """Auto-selecciona el JPG más nuevo al bootstrap o cuando la carpeta cambia.

    Tras `POST /media/clear` no re-selecciona archivos ya vistos (queda idle)
    hasta que aparezca/actualice un archivo nuevo.
    """
    while not stop.is_set():
        try:
            st = state()
            current = _scan_media_mtimes()
            if not st.media_watch_bootstrapped:
                st.media_watch_bootstrapped = True
                st.media_seen_mtimes = dict(current)
                if st.current_media is None:
                    newest = _pick_newest_name(current)
                    if newest:
                        _apply_media_selection(newest)
                        logger.info("Media watch bootstrap -> %s", newest)
            else:
                changed = _media_changes_detected(st.media_seen_mtimes, current)
                st.media_seen_mtimes = dict(current)
                if changed:
                    newest = _pick_newest_name(current)
                    if newest:
                        _apply_media_selection(newest)
                        logger.info("Media watch change -> %s", newest)
        except Exception as exc:
            logger.warning("Media watch tick failed: %s", exc)

        try:
            await asyncio.wait_for(stop.wait(), timeout=MEDIA_WATCH_INTERVAL)
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _app_state
    st = AppState()
    if _PLACEHOLDER_JPEG:
        st.latest_frame = _PLACEHOLDER_JPEG
    _app_state = st

    stop = asyncio.Event()
    sweeper = asyncio.create_task(_sweeper_loop(stop))
    watcher = asyncio.create_task(_media_watch_loop(stop))
    logger.info(
        "Adapter started TTL=%.1fs buffer=%d vi_env=%s media_watch=%.1fs",
        TRACK_TTL_SECONDS,
        EVENTS_BUFFER_SIZE,
        VI_ENV,
        MEDIA_WATCH_INTERVAL,
    )
    try:
        yield
    finally:
        stop.set()
        await sweeper
        await watcher
        _app_state = None


app = FastAPI(
    title="Vision Intelligence Adapter",
    description="Normaliza PaddleX → epp-core (schema 1.0)",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
# Spec: allow_origins=["*"] no puede ir con allow_credentials=True (browsers rechazan).
_cors_credentials = _cors_origins != ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Healthcheck para Docker Compose."""
    st = state()
    return {
        "status": "ok",
        "tracks_active": len(st.track_cache),
        "events_buffered": len(st.events_buffer),
        "paddlex_degraded": st.stats["paddlex_degraded"],
        "vi_env": VI_ENV,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/ingest")
async def ingest(request: Request) -> JSONResponse:
    """
    Recibe detecciones JSON de PaddleX (vía bridge) y las agrega al track_cache.

    Emisión (dos caminos, ambos pasan por `_sweep_expired_tracks` → `_flush_track`):
    - Foto activa (`current_media`): marca todos los tracks finalized y barre al
      instante (sin esperar TRACK_TTL). Es el camino del modo foto-only actual.
    - Sin foto / video futuro: el loop `_sweeper_loop` emite cuando expira el TTL
      o cuando una det marca `finalized`/`track_lost`. No es código muerto.
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

    raw_trace = body.get("trace_id", body.get("generation"))
    trace_id = None if raw_trace is None else str(raw_trace)
    token = _trace_id_var.set(trace_id)
    try:
        st = state()

        # El bridge correlaciona cada /ingest con la generación de foto activa
        # via trace_id (o generation) — si parsea como int, es la confirmación
        # de que esa generación fue efectivamente ingerida (usado por la SPA
        # para derivar completitud: generation == last_ingest_generation).
        if trace_id is not None:
            try:
                st.last_ingest_generation = int(trace_id)
            except (TypeError, ValueError):
                pass

        # Señal de degradación desde el bridge
        if body.get("degraded") is True:
            st.stats["paddlex_degraded"] = True
            return JSONResponse(
                {"ok": True, "degraded": True, "accepted": 0, "trace_id": trace_id}
            )

        detections = _extract_detections(body)
        if not detections:
            return JSONResponse(
                {
                    "ok": True,
                    "accepted": 0,
                    "message": "no detections",
                    "trace_id": trace_id,
                }
            )

        st.stats["paddlex_degraded"] = False
        accepted = 0
        for det in detections:
            track_id = det.get("track_id") or det.get("tracker_id")
            if track_id is None:
                continue
            _append_to_track(str(track_id), det)
            accepted += 1
            st.stats["ingested"] += 1

            # Finalización explícita (PaddleX / bridge puede marcar lost track)
            if det.get("finalized") or det.get("track_lost"):
                st.track_cache[str(track_id)].finalized = True

        # Foto activa: emitir ya (sin esperar TRACK_TTL).
        if st.current_media is not None:
            for tid in list(st.track_cache.keys()):
                st.track_cache[tid].finalized = True

        await _sweep_expired_tracks()

        return JSONResponse(
            {
                "ok": True,
                "accepted": accepted,
                "tracks_active": len(st.track_cache),
                "trace_id": trace_id,
            }
        )
    finally:
        _trace_id_var.reset(token)


@app.get("/events")
async def get_events(limit: int = 100, plate: Optional[str] = None) -> dict[str, Any]:
    """Buffer de PerceptionEvent para AMIS / ECharts.

    `plate` es un filtro de presentación (substring case-insensitive sobre
    `payload.plate_text`), aplicado ANTES de `limit`. Omitirlo preserva el
    comportamiento exacto previo. Distinto del matching `patente:` de
    rules-sink (candidate_ids) — no son intercambiables.
    """
    st = state()
    limit = max(1, min(limit, EVENTS_BUFFER_SIZE))
    source = list(st.events_buffer)
    if plate:
        needle = plate.lower()
        source = [
            e
            for e in source
            if needle in (e.get("payload", {}).get("plate_text") or "").lower()
        ]
    items = source[:limit]
    return {
        "count": len(items),
        "total_emitted": st.stats["emitted"],
        "tracks_active": len(st.track_cache),
        "degraded": st.stats["paddlex_degraded"],
        # Completitud derivada (SPA): generation == last_ingest_generation.
        "generation": st.generation,
        "last_ingest_generation": st.last_ingest_generation,
        "events": items,
    }


async def _relay_rules_sink(path: str, *, with_api_key: bool = False) -> dict[str, Any]:
    """GET read-only a rules-sink; nunca propaga error — {available:false} en su lugar."""
    try:
        import httpx

        headers = {}
        if with_api_key:
            headers["x-api-key"] = JETLINKS_API_KEY
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{RULES_SINK_URL}{path}", headers=headers)
            resp.raise_for_status()
            return {"available": True, **resp.json()}
    except Exception as exc:
        logger.warning("rules-sink unreachable at %s%s (%s)", RULES_SINK_URL, path, exc)
        return {"available": False}


@app.get("/rules/health")
async def rules_health() -> dict[str, Any]:
    """Relay read-only de rules-sink /health. Perfil `rules` off => {available:false}, nunca 5xx."""
    return await _relay_rules_sink("/health")


@app.get("/rules/alerts")
async def rules_alerts() -> dict[str, Any]:
    """Relay read-only de rules-sink /alerts (con x-api-key). Falla => {available:false}."""
    result = await _relay_rules_sink("/alerts", with_api_key=True)
    if not result.get("available"):
        return {"available": False, "count": 0, "alerts": []}
    return result


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


@app.get("/media/list")
async def media_list() -> dict[str, Any]:
    """Lista de muestras allow-listed disponibles (LMP-1)."""
    return {"items": _list_media_items()}


@app.get("/media/current")
async def media_current() -> dict[str, Any]:
    """Fuente activa actual + token de generación (LMP-2)."""
    st = state()
    if st.current_media is None:
        return {"name": None, "type": None, "generation": st.generation}
    return {
        "name": st.current_media["name"],
        "type": st.current_media["type"],
        "generation": st.generation,
    }


@app.get("/media/original")
async def media_original() -> Response:
    """Sirve el archivo original de la foto activa (sin overlay), para la SPA.

    `X-Generation` permite al cliente detectar que la foto activa cambió sin
    volver a pegar contra `/media/current`. `no-store`: la SPA nunca debe
    cachear la foto activa (cambia de nombre/contenido entre generaciones y
    un mismo nombre puede reaparecer tras un re-upload).
    404 si no hay foto activa o el archivo ya no está en el allow-list.
    """
    st = state()
    if st.current_media is None:
        return JSONResponse({"ok": False, "error": "no active media"}, status_code=404)

    resolved = _find_media_path(st.current_media["name"])
    if resolved is None:
        return JSONResponse({"ok": False, "error": "media not found"}, status_code=404)
    path, _media_type = resolved
    if not os.path.isfile(path):
        return JSONResponse({"ok": False, "error": "media not found"}, status_code=404)

    return FileResponse(
        path,
        headers={
            "X-Generation": str(st.generation),
            "Cache-Control": "no-store",
        },
    )


def _safe_upload_basename(original: str) -> Optional[str]:
    """Basename seguro con extensión de imagen; None si inválido."""
    if not original:
        return None
    base = os.path.basename(original.strip().replace("\\", "/"))
    if not base or base in (".", "..") or os.path.basename(base) != base:
        return None
    stem, ext = os.path.splitext(base)
    ext_l = ext.lower()
    if ext_l not in MEDIA_IMAGE_EXTENSIONS:
        return None
    stem_safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "upload"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{stem_safe}{ext_l}"


def _remember_media_mtime(name: str) -> None:
    """Actualiza el mapa del watcher tras un upload/select explícito."""
    path = os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR, name)
    try:
        state().media_seen_mtimes[name] = os.path.getmtime(path)
    except OSError:
        pass


@app.post("/media/select")
async def media_select(body: MediaSelectBody) -> JSONResponse:
    """Selecciona una foto allow-listed; rechaza video u otros nombres (LMP-1)."""
    result = _apply_media_selection(body.name)
    if result is None:
        return JSONResponse(
            {"ok": False, "error": "name not allow-listed (solo imagenes)"},
            status_code=400,
        )
    return JSONResponse(result)


@app.post("/media/upload")
async def media_upload(file: UploadFile = File(...)) -> JSONResponse:
    """Recibe una imagen, la guarda en MEDIA_DIR/images y la auto-selecciona.

    Respuesta compatible con AMIS `input-file` (status/msg/data) y con clients
    simples (`ok`/`name`/`generation`).
    """
    safe_name = _safe_upload_basename(file.filename or "")
    if safe_name is None:
        return JSONResponse(
            {
                "status": 1,
                "msg": "solo se aceptan imagenes (.jpg/.jpeg/.png/.bmp)",
                "ok": False,
                "error": "solo se aceptan imagenes (.jpg/.jpeg/.png/.bmp)",
            },
            status_code=400,
        )

    folder = os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR)
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        logger.error("Cannot create media folder %s: %s", folder, exc)
        return JSONResponse(
            {"status": 1, "msg": "no se pudo guardar", "ok": False, "error": str(exc)},
            status_code=500,
        )

    dest = os.path.join(folder, safe_name)
    try:
        data = await file.read()
        if not data:
            return JSONResponse(
                {
                    "status": 1,
                    "msg": "archivo vacio",
                    "ok": False,
                    "error": "archivo vacio",
                },
                status_code=400,
            )
        with open(dest, "wb") as fh:
            fh.write(data)
    except OSError as exc:
        logger.error("Upload write failed %s: %s", dest, exc)
        return JSONResponse(
            {"status": 1, "msg": "no se pudo guardar", "ok": False, "error": str(exc)},
            status_code=500,
        )
    finally:
        await file.close()

    _remember_media_mtime(safe_name)
    result = _apply_media_selection(safe_name)
    if result is None:
        return JSONResponse(
            {
                "status": 1,
                "msg": "guardado pero no seleccionable",
                "ok": False,
                "error": "guardado pero no seleccionable",
            },
            status_code=500,
        )

    logger.info(
        "Media uploaded: %s gen=%d trace_id=%s",
        safe_name,
        result["generation"],
        result["generation"],
    )
    return JSONResponse(
        {
            "status": 0,
            "msg": "",
            "ok": True,
            "name": safe_name,
            "generation": result["generation"],
            "data": {"value": safe_name, "name": safe_name},
        }
    )


@app.post("/media/clear")
async def media_clear() -> JSONResponse:
    """Quita la foto activa: el bridge pasa a idle hasta la próxima selección."""
    st = state()
    _flush_detection_session()
    st.current_media = None
    st.generation += 1
    logger.info(
        "Media cleared (gen=%d) — bridge idle until next photo trace_id=%s",
        st.generation,
        st.generation,
    )
    return JSONResponse({"ok": True, "name": None, "generation": st.generation})


@app.post("/preview/frame")
async def preview_frame(request: Request) -> JSONResponse:
    """Recibe el JPEG anotado del bridge y lo guarda como último frame (D1)."""
    st = state()
    body = await request.body()
    if not body:
        return JSONResponse({"ok": False, "error": "empty body"}, status_code=400)

    async with st.frame_lock:
        st.latest_frame = body
    return JSONResponse({"ok": True})


async def _mjpeg_frames(request: Request):
    """Generador multipart/x-mixed-replace: reemite el último frame disponible."""
    boundary = b"--frame"
    st = state()
    while True:
        if await request.is_disconnected():
            break
        async with st.frame_lock:
            frame = st.latest_frame
        if frame is None:
            await asyncio.sleep(PREVIEW_MJPEG_INTERVAL)
            continue
        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(frame)).encode("ascii")
            + b"\r\n\r\n"
            + frame
            + b"\r\n"
        )
        await asyncio.sleep(PREVIEW_MJPEG_INTERVAL)


@app.get("/preview.mjpg")
async def preview_mjpg(request: Request) -> StreamingResponse:
    """Stream MJPEG del preview anotado (heartbeat de la foto activa)."""
    return StreamingResponse(
        _mjpeg_frames(request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/preview.jpg")
async def preview_jpg() -> Response:
    """Único frame anotado más reciente, para foto (LMP-3)."""
    st = state()
    async with st.frame_lock:
        frame = st.latest_frame
    if frame is None:
        return Response(status_code=503, content=b"", media_type="image/jpeg")
    return Response(content=frame, media_type="image/jpeg")


@app.get("/")
async def dashboard() -> FileResponse:
    """Sirve el shell HTML que carga AMIS + amis_dashboard.json."""
    path = os.path.join(STATIC_DIR, "dashboard.html")
    return FileResponse(path)


# Assets estáticos (amis_dashboard.json, etc.)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
