"""
Puente resiliente: foto local o RTSP (MediaMTX) → PaddleX → Adapter.

PaddleX 3.x: pipeline `vehicle_attribute_recognition`
  POST /vehicle-attribute-recognition  { "image": "<base64>" }

PP-Vehicle (PaddleDetection) no existe en PaddleX 3. Este pipeline detecta
vehículos + atributos (color/tipo). No trae track_id: asignamos track_id con
IoU tracker mínimo (orquestación, no modelo propio).

Preview = overlay OpenCV local con etiquetas en inglés (tipo + color), sobre
el frame original. No usamos `result.image` de PaddleX (trae chino bilingüe).

OCR de patente (opcional, ENABLE_PLATE_OCR): tras escalar el bbox a coords
frame_hires, recorta y consulta un segundo servicio PaddleX (pipeline OCR,
`paddlex-ocr`) por detección elegible; merge en `d["plate"]`. Caída/timeout
del OCR deja `plate=None` sin degradar el pipeline attr primario.

Resiliencia: backoff RTSP (interruptible si hay foto seleccionada), degradación
si PaddleX cae, DEMO_MODE sin cámara. Muestras locales: solo imágenes bajo
MEDIA_DIR/images (no video de muestra).
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
# SOURCE_URL: RTSP vivo o ruta local de imagen. Sin setear → RTSP_URL.
# No hay rama de video de muestra local (solo fotos en MEDIA_DIR/images).
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
# Selector de muestra local (solo fotos): mismo layout que adapter.py.
MEDIA_DIR = os.getenv("MEDIA_DIR", "/media")
MEDIA_IMAGE_SUBDIR = "images"
MEDIA_POLL_INTERVAL = float(os.getenv("MEDIA_POLL_INTERVAL", "1.0"))
MEDIA_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
# Re-empuje del JPEG nativo en modo foto (single-shot) para /preview.mjpg.
PREVIEW_IMAGE_HEARTBEAT_SECONDS = float(
    os.getenv("PREVIEW_IMAGE_HEARTBEAT_SECONDS", "5.0")
)
PADDLEX_URL = os.getenv("PADDLEX_URL", "http://paddlex:8080")
PADDLEX_PREDICT_PATH = os.getenv(
    "PADDLEX_PREDICT_PATH", "/vehicle-attribute-recognition"
)
FPS = float(os.getenv("BRIDGE_FPS", "5"))
FRAME_INTERVAL = 1.0 / max(FPS, 0.1)
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
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

# --- Object detection COCO (servicio paddlex-objects, aditivo — ver docker-compose.yml) ---
# Corre en paralelo al pipeline vehicle_attribute_recognition; su caída nunca
# degrada el pipeline primario (mismo aislamiento que OCR, ver _infer_ocr).
PADDLEX_OBJECTS_URL = os.getenv("PADDLEX_OBJECTS_URL", "http://paddlex-objects:8082")
PADDLEX_OBJECTS_PREDICT_PATH = os.getenv(
    "PADDLEX_OBJECTS_PREDICT_PATH", "/object-detection"
)
# COCO labels que el pipeline vehicle_attribute_recognition ya cubre con más
# detalle (color/plate). Usado para deduplicar contra object_detection.
_VEHICLE_COCO_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}

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


def _iou(a: list[float], b: list[float]) -> float:
    """IoU standalone entre dos bboxes [x1,y1,x2,y2].

    Extraído de IoUTracker para reuso fuera del tracker (ver
    _merge_object_detections, dedupe vehicle/object detections).
    """
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


class IoUTracker:
    """Tracker mínimo por IoU para asignar track_id estables entre frames."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: dict[str, list[float]] = {}  # id -> bbox

    @staticmethod
    def _iou(a: list[float], b: list[float]) -> float:
        return _iou(a, b)

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
_object_tracker = IoUTracker(IOU_THRESHOLD)


def _is_rtsp_source(source: str) -> bool:
    """True si `source` es una URL RTSP; False si es una ruta de archivo local (BCS-1)."""
    return source.strip().lower().startswith("rtsp://")


def _media_type_by_extension(filename: str) -> Optional[str]:
    """Clasifica `filename` como 'image' por extensión; None si no matchea."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in MEDIA_IMAGE_EXTENSIONS:
        return "image"
    return None


def _is_safe_media_name(name: str) -> bool:
    """Allow-list guard local: `name` debe ser un basename plano (sin traversal)."""
    return bool(name) and os.path.basename(name) == name and name not in (".", "..")


def _resolve_media_path(name: str) -> Optional[str]:
    """Ruta absoluta bajo MEDIA_DIR/images para una muestra ya validada."""
    if not _is_safe_media_name(name):
        return None
    return os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR, name)


def _resolve_active_source(
    selected: Optional[dict[str, Any]],
) -> tuple[str, bool, bool]:
    """Decide la fuente activa: `(source, is_rtsp, is_image)`.

    Selector AMIS: solo fotos bajo MEDIA_DIR/images. Sin selección → SOURCE_URL
    si es RTSP o imagen; si SOURCE_URL apunta a otra cosa, cae a RTSP_URL.
    """
    if selected and selected.get("name"):
        media_type = selected.get("type") or "image"
        if media_type == "image" or _media_type_by_extension(selected["name"]) == "image":
            path = _resolve_media_path(selected["name"])
            if path is not None:
                return path, False, True
        logger.warning(
            "Selected media ignored (solo imagenes): name=%s type=%s",
            selected.get("name"),
            selected.get("type"),
        )
    if _is_rtsp_source(SOURCE_URL):
        return SOURCE_URL, True, False
    if _media_type_by_extension(SOURCE_URL) == "image":
        return SOURCE_URL, False, True
    logger.warning(
        "SOURCE_URL no es RTSP ni imagen; fallback a RTSP_URL=%s", RTSP_URL
    )
    return RTSP_URL, True, False


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


def _preview_label(det: dict[str, Any]) -> str:
    """Texto de etiqueta en inglés para el overlay (sin CJK)."""
    parts: list[str] = []
    label = det.get("label")
    color = det.get("color")
    if label:
        parts.append(str(label))
    if color:
        parts.append(str(color))
    return " ".join(parts) or "vehicle"


# Paleta BGR distinta por tipo (OpenCV); fallback por track_id.
_PREVIEW_TYPE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "sedan": (255, 160, 0),      # azul claro
    "suv": (0, 140, 255),        # naranja
    "van": (200, 0, 200),        # magenta
    "truck": (0, 215, 255),      # amarillo
    "bus": (0, 0, 255),          # rojo
    "mpv": (255, 0, 128),        # rosa
    "pickup": (0, 200, 120),     # verde agua
    "hatchback": (180, 100, 0),  # azul oscuro
    "car": (255, 160, 0),
    "vehicle": (80, 80, 255),    # rojo suave
}

_PREVIEW_FALLBACK_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 160, 0),
    (0, 140, 255),
    (200, 0, 200),
    (0, 215, 255),
    (0, 0, 255),
    (255, 0, 128),
    (0, 200, 120),
    (180, 100, 0),
)


def _preview_box_color(det: dict[str, Any]) -> tuple[int, int, int]:
    """Color BGR estable por tipo de vehículo (o track_id si falta tipo)."""
    label = str(det.get("label") or "").strip().lower()
    if label in _PREVIEW_TYPE_COLORS_BGR:
        return _PREVIEW_TYPE_COLORS_BGR[label]
    tid = str(det.get("track_id") or "0")
    digits = "".join(ch for ch in tid if ch.isdigit()) or "0"
    idx = int(digits) % len(_PREVIEW_FALLBACK_PALETTE_BGR)
    return _PREVIEW_FALLBACK_PALETTE_BGR[idx]


def _draw_preview(frame, detections: list[dict[str, Any]]) -> Optional[bytes]:
    """Dibuja bboxes + labels EN con color por tipo y fondo de etiqueta legible."""
    canvas = frame.copy()
    for det in detections or []:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1 = int(round(float(bbox[0])))
        y1 = int(round(float(bbox[1])))
        x2 = int(round(float(bbox[2])))
        y2 = int(round(float(bbox[3])))
        color = _preview_box_color(det)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        text = _preview_label(det)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        label_y1 = max(0, y1 - th - baseline - pad * 2)
        label_y2 = max(th + baseline + pad * 2, y1)
        label_x2 = min(canvas.shape[1] - 1, x1 + tw + pad * 2)
        cv2.rectangle(
            canvas, (x1, label_y1), (label_x2, label_y2), color, thickness=-1
        )
        cv2.putText(
            canvas,
            text,
            (x1 + pad, label_y2 - baseline - pad),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return _encode_jpeg(canvas)


def _maybe_resize_for_infer(frame_hires) -> tuple[Any, float, float]:
    """Deriva frame_infer de frame_hires; downscale solo si excede BRIDGE_MAX_WIDTH.

    frame_hires se mantiene intacto (OCR/crop). Retorna (frame_infer, scale_x,
    scale_y) donde scale_* multiplica coordenadas infer -> hires.
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
                # Prefijo "v-": namespacing frente a object_detection ("o-"),
                # evita colisión de track_id como key en adapter.py TrackBucket.
                "track_id": f"v-{tid}",
                "label": m["label"],
                "score": m["score"],
                "color": m["color"],
                "bbox": m["bbox"],
                "plate": None,  # completado por OCR en run_loop si ENABLE_PLATE_OCR
                "frame_ts": now,
                "entity_type": "vehicle",
            }
        )
    return detections


def _normalize_object_detection_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Traduce respuesta object_detection (COCO, 80 clases) → dicts crudos.

    Serving API PaddleX object_detection:
      { "result": { "boxes": [ { "cls_id", "label", "score", "coordinate" } ] } }

    A diferencia de `_normalize_paddlex_result`, NO asigna track_id (lo hace
    el caller vía `_object_tracker`, ver `_run_detections`) ni agrega
    `color`/`plate`: una sola label + score por box (sin `_parse_attr_labels`,
    que asume vocabulario color/tipo de vehículo y mal-etiquetaría cualquier
    clase COCO como "vehicle" vía su fallback).
    """
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = result.get("boxes") or []
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "boxes" in item:
                boxes.extend(item.get("boxes") or [])

    detections: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
        label = box.get("label") or box.get("cls_name") or ""
        score = float(box.get("score") or box.get("det_score") or 0.0)
        detections.append(
            {
                "label": str(label),
                "score": score,
                "bbox": bbox,
                "entity_type": "object",
            }
        )
    return detections


def _merge_object_detections(
    vehicle_dets: list[dict[str, Any]],
    object_dets: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Dedupe: descarta detecciones de `object_dets` ya cubiertas por el pipeline
    vehicle_attribute_recognition (mismo auto detectado dos veces).

    Se descarta una entrada de `object_dets` solo si su label está en
    `_VEHICLE_COCO_LABELS` Y su bbox solapa (IoU > `iou_threshold`) alguna
    bbox de `vehicle_dets` — el pipeline vehicle gana esa caja (trae
    color/plate). Labels no-vehículo (person, dog, ...) se conservan siempre,
    sin importar el solapamiento.
    """
    if not object_dets:
        return []
    vehicle_boxes = [v["bbox"] for v in vehicle_dets if v.get("bbox")]
    kept: list[dict[str, Any]] = []
    for det in object_dets:
        label = str(det.get("label") or "").strip().lower()
        bbox = det.get("bbox")
        if label in _VEHICLE_COCO_LABELS and bbox and vehicle_boxes:
            if any(_iou(bbox, vb) > iou_threshold for vb in vehicle_boxes):
                continue
        kept.append(det)
    return kept


def _decode_paddlex_result_image(data: dict[str, Any]) -> Optional[bytes]:
    """Decodifica `result.image` (Base64) del serving PaddleX a bytes JPEG/PNG.

    None si falta el campo o el decode falla — el caller no debe inventar overlay.
    """
    result = data.get("result", data)
    if not isinstance(result, dict):
        return None
    image_b64 = result.get("image")
    if not image_b64 or not isinstance(image_b64, str):
        return None
    payload = image_b64.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as exc:
        logger.warning("PaddleX result.image Base64 decode failed: %s", exc)
        return None
    return raw or None


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
    """POST base64 a /vehicle-attribute-recognition → detecciones.

    Ignora `result.image` (etiquetas bilingües con chino); el preview se dibuja
    localmente en inglés sobre el frame original.
    """
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


async def _infer_object_detection(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST base64 a `{PADDLEX_OBJECTS_URL}{PADDLEX_OBJECTS_PREDICT_PATH}`.

    Aislado del pipeline vehicle_attribute_recognition (mismo patrón que
    `_infer_ocr`): cualquier excepción/timeout devuelve None sin llamar
    `_notify_degraded` — object_detection es una capa secundaria/aditiva, su
    caída nunca degrada el pipeline vehicle primario.
    """
    url = f"{PADDLEX_OBJECTS_URL.rstrip('/')}{PADDLEX_OBJECTS_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Object detection infer error (isolated, sin degradar): %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.debug("Object detection error: %s", data.get("errorMsg"))
        return None
    return _normalize_object_detection_result(data)


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
        "type": data.get("type") or "image",
        "generation": data.get("generation"),
    }


async def _push_preview_frame(client: httpx.AsyncClient, jpeg: bytes) -> None:
    """POST del JPEG de preview (overlay EN) al adapter. Aislado: falla en silencio."""
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
) -> tuple[Optional[list[dict[str, Any]]], bool, Optional[bytes]]:
    """Infiere vehicle_attribute_recognition + object_detection (COCO) + OCR.

    Devuelve (detections, degraded, preview_jpeg).

    `detections is None` = saltar frame (encode/PaddleX vehicle falló). `[]` =
    OK sin dets. object_detection es aditivo/aislado (igual patrón que OCR):
    su caída jamás degrada ni bloquea el frame — solo se pierde ese layer.
    `preview_jpeg` = overlay inglés local sobre frame_hires (o None si encode
    falla).
    """
    frame_infer, scale_x, scale_y = _maybe_resize_for_infer(frame_hires)
    jpeg = _encode_jpeg(frame_infer)
    if jpeg is None:
        return None, False, None

    # Vehicle attr (primario) y object detection (aditivo) corren en paralelo:
    # son independientes, ambos consultan el mismo jpeg.
    vehicle_detections, object_raw = await asyncio.gather(
        _infer_paddlex(client, jpeg),
        _infer_object_detection(client, jpeg),
    )
    if vehicle_detections is None:
        await _notify_degraded(client)
        return None, True, None

    _scale_detections(vehicle_detections, scale_x, scale_y)

    # === Object detection (COCO, aditivo) — track + dedupe contra vehicle ===
    if object_raw:
        object_boxes = [d["bbox"] for d in object_raw]
        object_track_ids = _object_tracker.assign(object_boxes)
        now = datetime.now(timezone.utc).isoformat()
        object_detections = [
            {
                # Prefijo "o-": namespacing frente a vehicle ("v-"), evita
                # colisión de track_id como key en adapter.py TrackBucket.
                "track_id": f"o-{tid}",
                "label": d["label"],
                "score": d["score"],
                "bbox": d["bbox"],
                "entity_type": "object",
                "frame_ts": now,
            }
            for tid, d in zip(object_track_ids, object_raw)
        ]
        _scale_detections(object_detections, scale_x, scale_y)
        detections = vehicle_detections + _merge_object_detections(
            vehicle_detections, object_detections
        )
    else:
        detections = vehicle_detections
    # ========================================================================

    # === OCR de patente (opcional; bbox ya en coords frame_hires) ===
    if ENABLE_PLATE_OCR and vehicle_detections:
        eligible = sorted(
            (d for d in vehicle_detections if d.get("score", 0.0) > OCR_MIN_SCORE),
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

    preview_jpeg = _draw_preview(frame_hires, detections)
    return detections, False, preview_jpeg


async def _run_image_source(
    client: httpx.AsyncClient, path: str, selected_name: Optional[str]
) -> None:
    """Fuente foto (single-shot): imread+infer una vez; preview = overlay EN.

    Heartbeat del mismo JPEG + poll de hot-swap. Retorna al cambiar
    la selección activa.
    """
    global _tracker, _object_tracker
    _tracker = IoUTracker(IOU_THRESHOLD)
    _object_tracker = IoUTracker(IOU_THRESHOLD)

    frame_hires = cv2.imread(path)
    if frame_hires is None:
        raise RuntimeError(f"Cannot read image source: {path}")

    detections, _degraded, preview_jpeg = await _run_detections(client, frame_hires)
    detections = detections or []
    if detections:
        await _post_json(client, ADAPTER_INGEST_URL, {"detections": detections})

    if preview_jpeg is not None:
        await _push_preview_frame(client, preview_jpeg)
    else:
        logger.warning(
            "Image source %s: sin preview (encode overlay falló)",
            path,
        )
    logger.info("Image source ready: %s detections=%d", path, len(detections))

    last_heartbeat = time.monotonic()
    while True:
        await asyncio.sleep(MEDIA_POLL_INTERVAL)
        polled = await _fetch_current_media(client)
        # None = clear / sin selección → salir a RTSP/SOURCE_URL.
        if polled is None:
            logger.info("Media cleared -> leave image source (%s)", selected_name)
            return
        if polled.get("name") != selected_name:
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
        "fps=%.1f preview=overlay_en ocr_enabled=%s ocr_url=%s",
        SOURCE_URL,
        MEDIA_DIR,
        PADDLEX_URL,
        PADDLEX_PREDICT_PATH,
        ADAPTER_INGEST_URL,
        DEMO_MODE,
        FPS,
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

                # Poll de selección de muestra antes de (re)abrir la fuente.
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

                # Solo RTSP vivo (ops). Sin archivo de video local ni overlay.
                cap = _open_capture(source, is_rtsp=True)
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open source: {source}")

                logger.info("Source connected: %s (rtsp=True)", source)
                backoff = BACKOFF_INITIAL
                last_infer = 0.0
                last_media_poll = time.monotonic()
                infer_frame_count = 0
                metrics_window_start = time.monotonic()
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
                            break

                    ok, frame_hires = cap.read()
                    if not ok or frame_hires is None:
                        raise RuntimeError("RTSP read failed")

                    now = time.monotonic()

                    if infer_task is not None and infer_task.done():
                        try:
                            detections, _degraded, preview_jpeg = infer_task.result()
                            infer_ms = (time.monotonic() - infer_started_at) * 1000.0
                            if detections is not None:
                                await _post_json(
                                    client,
                                    ADAPTER_INGEST_URL,
                                    {"detections": detections},
                                )
                                if preview_jpeg is not None:
                                    if (
                                        preview_push_task is None
                                        or preview_push_task.done()
                                    ):
                                        preview_push_task = asyncio.create_task(
                                            _push_preview_frame(client, preview_jpeg)
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
                                        "dets=%d preview=%s",
                                        infer_ms,
                                        effective_fps,
                                        len(detections),
                                        preview_jpeg is not None,
                                    )
                                    metrics_window_start = time.monotonic()
                        except asyncio.CancelledError:
                            pass
                        except Exception as exc:
                            logger.warning("Infer task failed: %s", exc)
                        infer_task = None

                    if infer_task is None and (now - last_infer) >= FRAME_INTERVAL:
                        last_infer = now
                        infer_started_at = now
                        infer_task = asyncio.create_task(
                            _run_detections(client, frame_hires.copy())
                        )

                    elapsed = time.monotonic() - tick_start
                    sleep_for = FRAME_INTERVAL - elapsed
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "Bridge error: %s — reconnect in %.1fs", exc, backoff
                )
                deadline = time.monotonic() + backoff
                interrupted = False
                while time.monotonic() < deadline:
                    polled = await _fetch_current_media(client)
                    if polled is not None:
                        selected = polled
                        _, _, is_image = _resolve_active_source(selected)
                        if is_image:
                            logger.info(
                                "Media selected during backoff -> %s (interrupt reconnect)",
                                selected.get("name"),
                            )
                            interrupted = True
                            break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    await asyncio.sleep(min(1.0, remaining))
                if interrupted:
                    backoff = BACKOFF_INITIAL
                else:
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
