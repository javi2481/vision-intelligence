"""
Puente resiliente: RTSP (MediaMTX) → PaddleX → Adapter.

PaddleX 3.x: pipeline `vehicle_attribute_recognition`
  POST /vehicle-attribute-recognition  { "image": "<base64>" }

PP-Vehicle (PaddleDetection) no existe en PaddleX 3. Este pipeline detecta
vehículos + atributos (color/tipo). No trae track_id: asignamos track_id con
IoU tracker mínimo (orquestación, no modelo propio).

OCR de patente (opcional, ENABLE_PLATE_OCR): tras escalar el bbox a coords
frame_hires, recorta y consulta un segundo servicio PaddleX (pipeline OCR,
`paddlex-ocr`) por detección elegible; merge en `d["plate"]`. Caída/timeout
del OCR deja `plate=None` sin degradar el pipeline attr primario.

Resiliencia: backoff RTSP, degradación si PaddleX cae, DEMO_MODE sin cámara.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional

import cv2
import httpx

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] bridge: %(message)s",
)
logger = logging.getLogger("rtsp_bridge")

RTSP_URL = os.getenv("RTSP_URL", "rtsp://mediamtx:8554/webcam")
ADAPTER_INGEST_URL = os.getenv(
    "ADAPTER_INGEST_URL", "http://adapter:8000/ingest"
)
PADDLEX_URL = os.getenv("PADDLEX_URL", "http://paddlex:8080")
PADDLEX_PREDICT_PATH = os.getenv(
    "PADDLEX_PREDICT_PATH", "/vehicle-attribute-recognition"
)
FPS = float(os.getenv("BRIDGE_FPS", "1"))
FRAME_INTERVAL = 1.0 / max(FPS, 0.1)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
BACKOFF_INITIAL = float(os.getenv("BACKOFF_INITIAL", "1.0"))
BACKOFF_MAX = float(os.getenv("BACKOFF_MAX", "30.0"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
# Ancho máximo de inferencia: sobre este umbral se deriva frame_infer
# (downscale) desde frame_hires; a la par o por debajo, pass-through sin copia.
BRIDGE_MAX_WIDTH = int(os.getenv("BRIDGE_MAX_WIDTH", "1280"))
# Cada N frames inferidos se emite una línea de métricas (infer_ms/encode_ms/fps).
BRIDGE_METRICS_EVERY = int(os.getenv("BRIDGE_METRICS_EVERY", "30"))

# --- OCR de patente (servicio paddlex-ocr, opcional — ver docker-compose.yml) ---
PADDLEX_OCR_URL = os.getenv("PADDLEX_OCR_URL", "http://paddlex-ocr:8081")
ENABLE_PLATE_OCR = os.getenv("ENABLE_PLATE_OCR", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
OCR_MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.7"))
OCR_EVERY_N_FRAMES = max(1, int(os.getenv("OCR_EVERY_N_FRAMES", "5")))
OCR_TOPK = max(1, int(os.getenv("OCR_TOPK", "3")))
OCR_HTTP_TIMEOUT = float(os.getenv("OCR_HTTP_TIMEOUT", "5"))
# Patente: 5-8 alfanuméricos tras normalizar (upper + descartar no-alnum).
PLATE_REGEX = re.compile(r"^[A-Z0-9]{5,8}$")
# Guard de crop degenerado: lado mínimo en px para intentar OCR.
_OCR_MIN_CROP_PX = 8


class IoUTracker:
    """Tracker mínimo por IoU para asignar track_id estables entre frames."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[str, list[float]] = {}  # id -> bbox

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def assign(self, boxes: list[list[float]]) -> list[str]:
        used: set[str] = set()
        ids: list[str] = []
        for bbox in boxes:
            best_id, best_iou = None, 0.0
            for tid, prev in self._tracks.items():
                if tid in used:
                    continue
                score = self._iou(bbox, prev)
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= self.iou_threshold:
                self._tracks[best_id] = bbox
                used.add(best_id)
                ids.append(best_id)
            else:
                tid = str(self._next_id)
                self._next_id += 1
                self._tracks[tid] = bbox
                used.add(tid)
                ids.append(tid)
        # Limpiar tracks no vistos este frame
        self._tracks = {tid: self._tracks[tid] for tid in used}
        return ids


_tracker = IoUTracker(IOU_THRESHOLD)


def _open_capture(url: str) -> cv2.VideoCapture:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _encode_jpeg(frame) -> Optional[bytes]:
    ok, buf = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not ok:
        return None
    return buf.tobytes()


def _maybe_resize_for_infer(frame_hires) -> tuple[Any, float, float]:
    """Deriva frame_infer de frame_hires; downscale solo si excede BRIDGE_MAX_WIDTH.

    frame_hires se mantiene intacto en todos los casos (para overlay/OCR futuro).
    Retorna (frame_infer, scale_x, scale_y) donde scale_* multiplica coordenadas
    infer -> hires; pass-through devuelve el mismo objeto sin copia y 1.0/1.0.
    """
    h, w = frame_hires.shape[:2]
    if w <= BRIDGE_MAX_WIDTH:
        return frame_hires, 1.0, 1.0
    new_w = BRIDGE_MAX_WIDTH
    new_h = max(1, round(h * new_w / w))
    frame_infer = cv2.resize(
        frame_hires, (new_w, new_h), interpolation=cv2.INTER_AREA
    )
    return frame_infer, w / new_w, h / new_h


def _scale_detections(
    dets: list[dict[str, Any]], scale_x: float, scale_y: float
) -> list[dict[str, Any]]:
    """Escala bbox de coords frame_infer -> frame_hires. Pass-through si 1.0/1.0.

    No toca ninguna otra clave del dict de detección.
    """
    if scale_x == 1.0 and scale_y == 1.0:
        return dets
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        d["bbox"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
    return dets


def _crop_bbox(frame, bbox: list[float]) -> Optional[Any]:
    """Recorta `frame` al bbox, clippeado a los bordes de la imagen.

    Devuelve None si el área resultante es degenerada (guard mínimo), en vez
    de encodear un crop inválido.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), w))
    y1 = max(0, min(int(round(y1)), h))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if (x2 - x1) < _OCR_MIN_CROP_PX or (y2 - y1) < _OCR_MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def _parse_plate(
    rec_texts: list[Any], rec_scores: list[Any]
) -> Optional[dict[str, Any]]:
    """Normaliza rec_texts/rec_scores del OCR y devuelve el match de mayor score.

    Normalización: upper + descartar no-alfanumérico. Filtro: regex de
    patente 5-8 caracteres (`PLATE_REGEX`). None si ningún texto matchea.
    """
    best: Optional[dict[str, Any]] = None
    for text, score in zip(rec_texts or [], rec_scores or []):
        normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        if not PLATE_REGEX.match(normalized):
            continue
        score_f = float(score or 0.0)
        if best is None or score_f > best["score"]:
            best = {"text": normalized, "score": score_f}
    return best


def _parse_attr_labels(labels: list[Any], scores: list[Any]) -> tuple[Optional[str], Optional[str]]:
    """Extrae color y vehicle_type de labels tipo 'red(红色)', 'sedan(轿车)'."""
    color, vtype = None, None
    color_keys = {
        "red", "blue", "green", "yellow", "white", "black", "brown",
        "silver", "grey", "gray", "orange", "purple", "gold",
    }
    type_keys = {
        "sedan", "suv", "van", "truck", "bus", "mpv", "pickup", "car",
    }
    for label, score in zip(labels or [], scores or []):
        raw = str(label).split("(")[0].strip().lower()
        if raw in color_keys and color is None:
            color = raw
        elif raw in type_keys and vtype is None:
            vtype = raw
        elif color is None and raw not in type_keys:
            # primer atributo no-tipo → color candidato
            color = raw
    return color, vtype


def _normalize_paddlex_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Traduce respuesta vehicle_attribute_recognition → dicts del adaptador.

    Serving API PaddleX 3.7 (HTTP):
      { "result": { "vehicles": [ { "bbox", "attributes", "score" } ], "image": "..." } }

    Predict local (docs):
      { "result": { "boxes": [ { labels, cls_scores, det_score, coordinate } ] } }
    """
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = (
            result.get("vehicles")
            or result.get("boxes")
            or result.get("detections")
            or []
        )
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and (
                "boxes" in item or "vehicles" in item
            ):
                boxes.extend(item.get("boxes") or item.get("vehicles") or [])
            elif isinstance(item, dict) and (
                "coordinate" in item or "bbox" in item
            ):
                boxes.append(item)

    coords: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]

        # Formato serving: attributes = [{label, score}, ...] o labels planos
        labels: list[Any] = []
        scores: list[Any] = []
        attrs = box.get("attributes")
        if isinstance(attrs, list):
            for a in attrs:
                if isinstance(a, dict):
                    labels.append(a.get("label") or a.get("name") or "")
                    scores.append(a.get("score") or a.get("confidence") or 0.0)
                else:
                    labels.append(a)
                    scores.append(1.0)
        else:
            labels = list(box.get("labels") or [])
            scores = list(box.get("cls_scores") or [])

        color, vtype = _parse_attr_labels(labels, scores)
        coords.append(bbox)
        meta.append(
            {
                "label": vtype or "vehicle",
                "score": float(
                    box.get("score")
                    or box.get("det_score")
                    or 0.0
                ),
                "color": color,
                "bbox": bbox,
            }
        )

    track_ids = _tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    detections: list[dict[str, Any]] = []
    for tid, m in zip(track_ids, meta):
        detections.append(
            {
                "track_id": tid,
                "label": m["label"],
                "score": m["score"],
                "color": m["color"],
                "bbox": m["bbox"],
                "plate": None,  # completado por OCR en run_loop si ENABLE_PLATE_OCR
                "frame_ts": now,
            }
        )
    return detections


def _demo_detections() -> list[dict[str, Any]]:
    track_id = str(random.randint(1, 5))
    plates = ["ABC123", "ABC123", "ABG123", "XYZ789", "ABC123"]
    colors = ["white", "white", "silver", "black", "white"]
    return [
        {
            "track_id": track_id,
            "label": random.choice(["car", "truck", "bus"]),
            "score": round(random.uniform(0.6, 0.98), 3),
            "color": random.choice(colors),
            "bbox": [100, 120, 340, 280],
            "plate": {
                "text": random.choice(plates),
                "score": round(random.uniform(0.5, 0.97), 3),
            },
            "frame_ts": datetime.now(timezone.utc).isoformat(),
        }
    ]


async def _post_json(client: httpx.AsyncClient, url: str, payload: Any) -> bool:
    try:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return True
    except Exception as exc:
        logger.warning("POST %s failed: %s", url, exc)
        return False


async def _infer_paddlex(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST base64 a /vehicle-attribute-recognition y normaliza a detecciones."""
    url = f"{PADDLEX_URL.rstrip('/')}{PADDLEX_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("PaddleX infer error: %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.warning("PaddleX error: %s", data.get("errorMsg"))
        return None
    return _normalize_paddlex_result(data)


async def _infer_ocr(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[dict[str, Any]]:
    """POST base64 a `{PADDLEX_OCR_URL}/ocr` y parsea la mejor patente.

    Aislado del pipeline attr (D4 en design): cualquier excepción/timeout
    devuelve None sin llamar `_notify_degraded` — OCR caído no degrada el
    bridge globalmente, solo deja `plate=None` para esa detección.
    """
    url = f"{PADDLEX_OCR_URL.rstrip('/')}/ocr"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"file": b64, "fileType": 1}, timeout=OCR_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("OCR infer error (isolated, sin degradar): %s", exc)
        return None

    try:
        result = data.get("result", data) if isinstance(data, dict) else {}
        ocr_results = result.get("ocrResults") or []
        if not ocr_results:
            return None
        pruned = ocr_results[0].get("prunedResult") or {}
        rec_texts = pruned.get("rec_texts") or []
        rec_scores = pruned.get("rec_scores") or []
    except Exception as exc:
        logger.debug("OCR result parse error: %s", exc)
        return None
    return _parse_plate(rec_texts, rec_scores)


async def _notify_degraded(client: httpx.AsyncClient) -> None:
    await _post_json(client, ADAPTER_INGEST_URL, {"degraded": True})


async def run_loop() -> None:
    backoff = BACKOFF_INITIAL
    logger.info(
        "Bridge start rtsp=%s paddlex=%s%s adapter=%s demo=%s fps=%.1f "
        "ocr_enabled=%s ocr_url=%s",
        RTSP_URL,
        PADDLEX_URL,
        PADDLEX_PREDICT_PATH,
        ADAPTER_INGEST_URL,
        DEMO_MODE,
        FPS,
        ENABLE_PLATE_OCR,
        PADDLEX_OCR_URL if ENABLE_PLATE_OCR else "-",
    )

    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        while True:
            cap: Optional[cv2.VideoCapture] = None
            try:
                if DEMO_MODE:
                    detections = _demo_detections()
                    await _post_json(
                        client, ADAPTER_INGEST_URL, {"detections": detections}
                    )
                    await asyncio.sleep(FRAME_INTERVAL)
                    backoff = BACKOFF_INITIAL
                    continue

                cap = _open_capture(RTSP_URL)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open RTSP: {RTSP_URL}")

                logger.info("RTSP connected: %s", RTSP_URL)
                backoff = BACKOFF_INITIAL
                last_sent = 0.0
                infer_frame_count = 0
                ocr_frame_idx = 0
                metrics_window_start = time.monotonic()

                while True:
                    ok, frame_hires = cap.read()
                    if not ok or frame_hires is None:
                        raise RuntimeError("RTSP read failed")

                    now = time.monotonic()
                    if now - last_sent < FRAME_INTERVAL:
                        await asyncio.sleep(0.01)
                        continue
                    last_sent = now

                    frame_infer, scale_x, scale_y = _maybe_resize_for_infer(
                        frame_hires
                    )
                    resized = not (scale_x == 1.0 and scale_y == 1.0)

                    encode_start = time.monotonic()
                    jpeg = _encode_jpeg(frame_infer)
                    encode_ms = (time.monotonic() - encode_start) * 1000.0
                    if jpeg is None:
                        continue

                    infer_start = time.monotonic()
                    detections = await _infer_paddlex(client, jpeg)
                    infer_ms = (time.monotonic() - infer_start) * 1000.0
                    if detections is None:
                        await _notify_degraded(client)
                        continue

                    _scale_detections(detections, scale_x, scale_y)

                    # === OCR de patente (opcional, D1: bbox ya en coords
                    # frame_hires tras _scale_detections — sin 1/scale extra) ===
                    if ENABLE_PLATE_OCR:
                        ocr_frame_idx += 1
                        if ocr_frame_idx % OCR_EVERY_N_FRAMES == 0:
                            eligible = sorted(
                                (
                                    d
                                    for d in detections
                                    if d.get("score", 0.0) > OCR_MIN_SCORE
                                ),
                                key=lambda d: d["score"],
                                reverse=True,
                            )[:OCR_TOPK]
                            for d in eligible:
                                crop = _crop_bbox(frame_hires, d["bbox"])
                                if crop is None:
                                    continue
                                crop_jpeg = _encode_jpeg(crop)
                                if crop_jpeg is None:
                                    continue
                                plate = await _infer_ocr(client, crop_jpeg)
                                if plate:
                                    d["plate"] = plate
                    # ============================================================

                    await _post_json(
                        client,
                        ADAPTER_INGEST_URL,
                        {"detections": detections},
                    )

                    infer_frame_count += 1
                    if infer_frame_count % BRIDGE_METRICS_EVERY == 0:
                        window_s = time.monotonic() - metrics_window_start
                        effective_fps = (
                            BRIDGE_METRICS_EVERY / window_s if window_s > 0 else 0.0
                        )
                        logger.info(
                            "metrics infer_ms=%.1f encode_ms=%.1f "
                            "effective_fps=%.2f resized=%s infer_w=%d",
                            infer_ms,
                            encode_ms,
                            effective_fps,
                            resized,
                            frame_infer.shape[1],
                        )
                        metrics_window_start = time.monotonic()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Bridge error: %s — reconnect in %.1fs", exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, BACKOFF_MAX)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass


def main() -> None:
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        logger.info("Bridge stopped")


if __name__ == "__main__":
    main()
