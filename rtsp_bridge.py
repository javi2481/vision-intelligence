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
# Evita spam/latencia: httpx loguea cada POST /preview.frame a INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

RTSP_URL = os.getenv("RTSP_URL", "rtsp://mediamtx:8554/webcam")
# SOURCE_URL generaliza el origen: RTSP (rtsp://...) o ruta local de archivo
# bajo MEDIA_DIR. Sin setear, cae a RTSP_URL (compat total con el default
# previo — BCS-1).
SOURCE_URL = os.getenv("SOURCE_URL") or RTSP_URL
ADAPTER_INGEST_URL = os.getenv(
    "ADAPTER_INGEST_URL", "http://adapter:8000/ingest"
)
ADAPTER_MEDIA_CURRENT_URL = os.getenv(
    "ADAPTER_MEDIA_CURRENT_URL", "http://adapter:8000/media/current"
)
ADAPTER_PREVIEW_FRAME_URL = os.getenv(
    "ADAPTER_PREVIEW_FRAME_URL", "http://adapter:8000/preview/frame"
)
# Selector de muestra local (overlay-preview): mismo layout que adapter.py.
MEDIA_DIR = os.getenv("MEDIA_DIR", "/media")
MEDIA_VIDEO_SUBDIR = "videos"
MEDIA_IMAGE_SUBDIR = "images"
MEDIA_POLL_INTERVAL = float(os.getenv("MEDIA_POLL_INTERVAL", "1.0"))
MEDIA_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MEDIA_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# Cada cuántos segundos se re-empuja el mismo frame anotado en modo foto
# (single-shot) para mantener /preview.mjpg "vivo" en el adaptador (D5).
PREVIEW_IMAGE_HEARTBEAT_SECONDS = float(
    os.getenv("PREVIEW_IMAGE_HEARTBEAT_SECONDS", "5.0")
)
# Overlay (FO-1): color BGR y fuente del bbox+label dibujado sobre frame_hires.
OVERLAY_BOX_COLOR = (0, 255, 0)
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_FONT_SCALE = 0.5
PADDLEX_URL = os.getenv("PADDLEX_URL", "http://paddlex:8080")
PADDLEX_PREDICT_PATH = os.getenv(
    "PADDLEX_PREDICT_PATH", "/vehicle-attribute-recognition"
)
FPS = float(os.getenv("BRIDGE_FPS", "5"))
FRAME_INTERVAL = 1.0 / max(FPS, 0.1)
# Preview fluido: ritmo de lectura/push al dashboard, independiente de la
# inferencia. Los recuadros se dibujan sobre el frame ACTUAL (puede haber un
# leve atraso vs la detección; no se reinyecta el frame viejo — eso entrecorta).
PREVIEW_FPS = float(os.getenv("PREVIEW_FPS", "20"))
PREVIEW_INTERVAL = 1.0 / max(PREVIEW_FPS, 0.1)
# Ancho máx. del JPEG de preview (más chico = encode/red más fluido).
PREVIEW_ENCODE_MAX_WIDTH = int(os.getenv("PREVIEW_ENCODE_MAX_WIDTH", "960"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
JPEG_QUALITY_PREVIEW = int(os.getenv("JPEG_QUALITY_PREVIEW", "60"))
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
BACKOFF_INITIAL = float(os.getenv("BACKOFF_INITIAL", "1.0"))
BACKOFF_MAX = float(os.getenv("BACKOFF_MAX", "30.0"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
# Ancho máximo de inferencia: sobre este umbral se deriva frame_infer
# (downscale) desde frame_hires; a la par o por debajo, pass-through sin copia.
BRIDGE_MAX_WIDTH = int(os.getenv("BRIDGE_MAX_WIDTH", "960"))
# Cada N frames inferidos se emite una línea de métricas (infer_ms/fps/dets).
BRIDGE_METRICS_EVERY = int(os.getenv("BRIDGE_METRICS_EVERY", "30"))
# Compensa el atraso del recuadro: proyecta bbox con velocidad del track
# hasta el instante del frame actual (máx. segundos hacia adelante).
OVERLAY_EXTRAPOLATE_MAX_SEC = float(os.getenv("OVERLAY_EXTRAPOLATE_MAX_SEC", "0.6"))

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


def _is_rtsp_source(source: str) -> bool:
    """True si `source` es una URL RTSP; False si es una ruta de archivo local (BCS-1)."""
    return source.strip().lower().startswith("rtsp://")


def _media_type_by_extension(filename: str) -> Optional[str]:
    """Clasifica `filename` en 'video'/'image' por extensión; None si no matchea."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in MEDIA_VIDEO_EXTENSIONS:
        return "video"
    if ext in MEDIA_IMAGE_EXTENSIONS:
        return "image"
    return None


def _is_safe_media_name(name: str) -> bool:
    """Allow-list guard local: `name` debe ser un basename plano (sin traversal)."""
    return bool(name) and os.path.basename(name) == name and name not in (".", "..")


def _resolve_media_path(name: str, media_type: str) -> Optional[str]:
    """Construye la ruta absoluta bajo MEDIA_DIR para una muestra ya validada
    por el adapter. Vuelve a chequear traversal (defensa en profundidad)."""
    if not _is_safe_media_name(name):
        return None
    subdir = MEDIA_VIDEO_SUBDIR if media_type == "video" else MEDIA_IMAGE_SUBDIR
    return os.path.join(MEDIA_DIR, subdir, name)


def _resolve_active_source(
    selected: Optional[dict[str, Any]],
) -> tuple[str, bool, bool]:
    """Decide la fuente activa: `(source, is_rtsp, is_image)`.

    Si hay una muestra seleccionada (vía /media/select en el adapter), resuelve
    su ruta local bajo MEDIA_DIR. Si no, cae a SOURCE_URL (RTSP o archivo local
    directo) — comportamiento previo preservado cuando no hay selección.
    """
    if selected and selected.get("name"):
        media_type = selected.get("type") or "video"
        path = _resolve_media_path(selected["name"], media_type)
        if path is not None:
            return path, False, media_type == "image"
    is_rtsp = _is_rtsp_source(SOURCE_URL)
    is_image = (not is_rtsp) and _media_type_by_extension(SOURCE_URL) == "image"
    return SOURCE_URL, is_rtsp, is_image


def _open_capture(source: str, is_rtsp: bool = True) -> cv2.VideoCapture:
    if is_rtsp:
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _encode_jpeg(frame, quality: Optional[int] = None) -> Optional[bytes]:
    q = JPEG_QUALITY if quality is None else quality
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return None
    return buf.tobytes()


def _encode_preview_jpeg(frame) -> Optional[bytes]:
    """JPEG liviano para el panel: downscale + calidad preview (fluidez)."""
    h, w = frame.shape[:2]
    if PREVIEW_ENCODE_MAX_WIDTH > 0 and w > PREVIEW_ENCODE_MAX_WIDTH:
        new_w = PREVIEW_ENCODE_MAX_WIDTH
        new_h = max(1, round(h * new_w / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return _encode_jpeg(frame, quality=JPEG_QUALITY_PREVIEW)


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


# Etiquetas visibles en overlay (español). Claves = valores crudos del modelo.
_OVERLAY_LABEL_ES = {
    "vehicle": "vehiculo",
    "car": "auto",
    "sedan": "sedan",
    "suv": "SUV",
    "van": "furgoneta",
    "truck": "camion",
    "bus": "colectivo",
    "mpv": "monovolumen",
    "pickup": "pickup",
    "unknown": "desconocido",
}


def _overlay_type_es(raw: Any) -> str:
    """Tipo para dibujar en el frame: español ASCII (OpenCV putText no rinde bien tildes)."""
    key = re.sub(r"[\u4e00-\u9fff]+", "", str(raw or "vehicle"))
    key = key.split("(")[0].strip().lower() or "vehicle"
    return _OVERLAY_LABEL_ES.get(key, key)


def _extrapolate_detections(
    curr: list[dict[str, Any]],
    curr_ts: float,
    prev: Optional[list[dict[str, Any]]],
    prev_ts: float,
    now: float,
    max_ahead: float = OVERLAY_EXTRAPOLATE_MAX_SEC,
) -> list[dict[str, Any]]:
    """Proyecta bboxes al instante `now` usando velocidad constante por track_id.

    `curr_ts` debe ser el timestamp del FOTOGRAMA inferido (no el de fin de
    inferencia): si PaddleX tarda 0.4s, sin esto el recuadro ya nace atrasado.
    """
    if not curr:
        return curr
    dt_ahead = now - curr_ts
    if dt_ahead <= 0.001 or not prev or curr_ts <= prev_ts:
        return curr
    dt_hist = curr_ts - prev_ts
    if dt_hist < 0.02:
        return curr
    dt_ahead = min(float(dt_ahead), float(max_ahead))
    prev_by = {
        str(d.get("track_id")): d
        for d in prev
        if d.get("track_id") is not None and d.get("bbox")
    }
    out: list[dict[str, Any]] = []
    for d in curr:
        nd = dict(d)
        bbox = d.get("bbox")
        p = prev_by.get(str(d.get("track_id")))
        pb = p.get("bbox") if p else None
        if (
            bbox
            and pb
            and len(bbox) >= 4
            and len(pb) >= 4
        ):
            pred = []
            for i in range(4):
                v = (float(bbox[i]) - float(pb[i])) / dt_hist
                pred.append(float(bbox[i]) + v * dt_ahead)
            nd["bbox"] = pred
        out.append(nd)
    return out


def _draw_overlay(frame, dets: list[dict[str, Any]]):
    """Dibuja bbox+label (tipo/patente/confianza) por detección sobre una copia
    de `frame` (FO-1). Sin detecciones, devuelve `frame` sin tocar (no-op)."""
    if not dets:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    for d in dets:
        bbox = d.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1 = max(0, min(int(round(bbox[0])), w))
        y1 = max(0, min(int(round(bbox[1])), h))
        x2 = max(0, min(int(round(bbox[2])), w))
        y2 = max(0, min(int(round(bbox[3])), h))
        if x2 <= x1 or y2 <= y1:
            continue

        plate = d.get("plate")
        plate_text = plate.get("text") if isinstance(plate, dict) else None
        label = _overlay_type_es(d.get("label"))
        if plate_text:
            label += f" {plate_text}"
        label += f" {float(d.get('score') or 0.0):.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), OVERLAY_BOX_COLOR, 2)
        (tw, th), _ = cv2.getTextSize(label, OVERLAY_FONT, OVERLAY_FONT_SCALE, 1)
        ty1 = max(0, y1 - th - 6)
        cv2.rectangle(annotated, (x1, ty1), (x1 + tw + 4, ty1 + th + 6), OVERLAY_BOX_COLOR, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, ty1 + th + 2),
            OVERLAY_FONT,
            OVERLAY_FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


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
    """Extrae color y vehicle_type de labels tipo 'red(红色)', 'sedan(轿车)'.

    Solo conserva la parte latina antes del paréntesis; descarta CJK residual
    para que color/tipo no lleguen en chino al front.
    """
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
        raw = re.sub(r"[\u4e00-\u9fff]+", "", raw).strip()
        if not raw:
            continue
        if raw in color_keys and color is None:
            color = raw
        elif raw in type_keys and vtype is None:
            vtype = raw
        elif color is None and raw not in type_keys and re.fullmatch(r"[a-z_\-]+", raw):
            # primer atributo latino no-tipo → color candidato (nunca CJK)
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


async def _fetch_current_media(client: httpx.AsyncClient) -> Optional[dict[str, Any]]:
    """GET /media/current en el adapter. None ante cualquier falla (D3: se
    mantiene la selección activa en silencio, sin backoff-reconnect)."""
    try:
        resp = await client.get(ADAPTER_MEDIA_CURRENT_URL, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Media poll failed (kept active source): %s", exc)
        return None

    if not isinstance(data, dict) or not data.get("name"):
        return None
    return {
        "name": data["name"],
        "type": data.get("type") or "video",
        "generation": data.get("generation"),
    }


async def _push_preview_frame(client: httpx.AsyncClient, jpeg: bytes) -> None:
    """POST del JPEG anotado al adapter (D1). Aislado: falla en silencio, sin
    afectar el output /ingest (LMP-4, dual output)."""
    try:
        resp = await client.post(
            ADAPTER_PREVIEW_FRAME_URL,
            content=jpeg,
            headers={"Content-Type": "image/jpeg"},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
    except Exception as exc:
        logger.debug("Preview push failed: %s", exc)


async def _run_detections(
    client: httpx.AsyncClient, frame_hires
) -> tuple[Optional[list[dict[str, Any]]], bool]:
    """Infiere+OCR sobre un frame ya capturado. Devuelve (detections, degraded).

    `detections is None` señala "saltar este frame" (falla de encode o
    PaddleX degradado) — distinto de `[]` (inferencia OK, cero detecciones).
    Compartido entre el loop de video y el modo foto single-shot: mismo
    pipeline resize -> encode -> PaddleX -> scale -> OCR opcional.
    """
    frame_infer, scale_x, scale_y = _maybe_resize_for_infer(frame_hires)
    jpeg = _encode_jpeg(frame_infer)
    if jpeg is None:
        return None, False

    detections = await _infer_paddlex(client, jpeg)
    if detections is None:
        await _notify_degraded(client)
        return None, True

    _scale_detections(detections, scale_x, scale_y)

    # === OCR de patente (opcional, D1: bbox ya en coords frame_hires tras
    # _scale_detections — sin 1/scale extra) ===
    if ENABLE_PLATE_OCR and detections:
        eligible = sorted(
            (d for d in detections if d.get("score", 0.0) > OCR_MIN_SCORE),
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

    return detections, False


async def _run_image_source(
    client: httpx.AsyncClient, path: str, selected_name: Optional[str]
) -> None:
    """Fuente foto (single-shot, D5): imread+infer+overlay una vez, luego
    idle re-empujando el mismo frame anotado + polleando hot-swap.

    Retorna cuando la selección activa cambia (para que run_loop reabra).
    """
    frame_hires = cv2.imread(path)
    if frame_hires is None:
        raise RuntimeError(f"Cannot read image source: {path}")

    detections, _degraded = await _run_detections(client, frame_hires)
    detections = detections or []
    if detections:
        await _post_json(client, ADAPTER_INGEST_URL, {"detections": detections})

    annotated = _draw_overlay(frame_hires, detections)
    preview_jpeg = _encode_preview_jpeg(annotated)
    if preview_jpeg is not None:
        await _push_preview_frame(client, preview_jpeg)
    logger.info("Image source ready: %s detections=%d", path, len(detections))

    last_heartbeat = time.monotonic()
    while True:
        await asyncio.sleep(MEDIA_POLL_INTERVAL)
        polled = await _fetch_current_media(client)
        if polled is not None and polled.get("name") != selected_name:
            logger.info("Media selection changed away from image -> %s", polled)
            return

        now = time.monotonic()
        if preview_jpeg is not None and now - last_heartbeat >= PREVIEW_IMAGE_HEARTBEAT_SECONDS:
            await _push_preview_frame(client, preview_jpeg)
            last_heartbeat = now


async def run_loop() -> None:
    backoff = BACKOFF_INITIAL
    logger.info(
        "Bridge start source=%s media_dir=%s paddlex=%s%s adapter=%s demo=%s "
        "fps=%.1f preview_fps=%.1f ocr_enabled=%s ocr_url=%s",
        SOURCE_URL,
        MEDIA_DIR,
        PADDLEX_URL,
        PADDLEX_PREDICT_PATH,
        ADAPTER_INGEST_URL,
        DEMO_MODE,
        FPS,
        PREVIEW_FPS,
        ENABLE_PLATE_OCR,
        PADDLEX_OCR_URL if ENABLE_PLATE_OCR else "-",
    )

    selected: Optional[dict[str, Any]] = None

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

                # Poll de selección de muestra antes de (re)abrir la fuente (LMP-2, D3).
                polled = await _fetch_current_media(client)
                if polled is not None:
                    selected = polled
                source, is_rtsp, is_image = _resolve_active_source(selected)

                if is_image:
                    await _run_image_source(
                        client, source, selected.get("name") if selected else None
                    )
                    backoff = BACKOFF_INITIAL
                    continue

                cap = _open_capture(source, is_rtsp)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open source: {source}")

                logger.info("Source connected: %s (rtsp=%s)", source, is_rtsp)
                backoff = BACKOFF_INITIAL
                last_infer = 0.0
                last_media_poll = time.monotonic()
                infer_frame_count = 0
                metrics_window_start = time.monotonic()
                last_detections: list[dict[str, Any]] = []
                prev_detections: list[dict[str, Any]] = []
                curr_det_ts = 0.0
                prev_det_ts = 0.0
                infer_task: Optional[asyncio.Task] = None
                infer_started_at = 0.0
                preview_push_task: Optional[asyncio.Task] = None

                while True:
                    tick_start = time.monotonic()
                    now_poll = tick_start
                    if now_poll - last_media_poll >= MEDIA_POLL_INTERVAL:
                        last_media_poll = now_poll
                        polled = await _fetch_current_media(client)
                        active_name = selected.get("name") if selected else None
                        if polled is not None and polled.get("name") != active_name:
                            logger.info("Media selection changed -> %s", polled)
                            selected = polled
                            if infer_task is not None and not infer_task.done():
                                infer_task.cancel()
                            break  # reabrir con la nueva fuente en la vuelta externa

                    ok, frame_hires = cap.read()
                    if not ok or frame_hires is None:
                        if is_rtsp:
                            raise RuntimeError("RTSP read failed")
                        # EOF en archivo local: rewind, NO backoff-reconnect (BCS-2).
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame_hires = cap.read()
                        if not ok or frame_hires is None:
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = _open_capture(source, is_rtsp)
                            if not cap.isOpened():
                                raise RuntimeError(
                                    f"Cannot reopen local source after EOF: {source}"
                                )
                            ok, frame_hires = cap.read()
                            if not ok or frame_hires is None:
                                raise RuntimeError(
                                    f"Local source unreadable after reopen: {source}"
                                )
                        logger.debug("Local source EOF -> rewound: %s", source)

                    now = time.monotonic()

                    # Recolectar inferencia en background (no bloquea el preview).
                    if infer_task is not None and infer_task.done():
                        try:
                            detections, _degraded = infer_task.result()
                            infer_ms = (time.monotonic() - infer_started_at) * 1000.0
                            if detections is not None:
                                prev_detections = last_detections
                                prev_det_ts = curr_det_ts
                                last_detections = detections
                                # Timestamp del FOTOGRAMA analizado, no del fin
                                # de inferencia (compensa los ~400ms de PaddleX).
                                curr_det_ts = infer_started_at
                                await _post_json(
                                    client,
                                    ADAPTER_INGEST_URL,
                                    {"detections": detections},
                                )
                                infer_frame_count += 1
                                if infer_frame_count % BRIDGE_METRICS_EVERY == 0:
                                    window_s = time.monotonic() - metrics_window_start
                                    effective_fps = (
                                        BRIDGE_METRICS_EVERY / window_s
                                        if window_s > 0
                                        else 0.0
                                    )
                                    logger.info(
                                        "metrics infer_ms=%.1f effective_fps=%.2f "
                                        "preview_fps=%.1f dets=%d",
                                        infer_ms,
                                        effective_fps,
                                        PREVIEW_FPS,
                                        len(detections),
                                    )
                                    metrics_window_start = time.monotonic()
                        except asyncio.CancelledError:
                            pass
                        except Exception as exc:
                            logger.warning("Infer task failed: %s", exc)
                        infer_task = None

                    # Lanzar nueva inferencia solo al ritmo BRIDGE_FPS.
                    if infer_task is None and (now - last_infer) >= FRAME_INTERVAL:
                        last_infer = now
                        infer_started_at = now
                        infer_task = asyncio.create_task(
                            _run_detections(client, frame_hires.copy())
                        )

                    # Solo frames ACTUALES; bbox proyectado al instante actual.
                    dets_draw = _extrapolate_detections(
                        last_detections,
                        curr_det_ts,
                        prev_detections,
                        prev_det_ts,
                        now,
                    )
                    annotated = _draw_overlay(frame_hires, dets_draw)
                    preview_jpeg = _encode_preview_jpeg(annotated)
                    if preview_jpeg is not None:
                        # Si el push anterior sigue en vuelo, saltar este frame
                        # en vez de acumular cola (también entrecorta).
                        if preview_push_task is None or preview_push_task.done():
                            preview_push_task = asyncio.create_task(
                                _push_preview_frame(client, preview_jpeg)
                            )

                    elapsed = time.monotonic() - tick_start
                    sleep_for = PREVIEW_INTERVAL - elapsed
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
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
