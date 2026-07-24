"""Cliente HTTP al pipeline vehicle_attribute_recognition (PaddleX :8080).

Recibe JPEG de inferencia; devuelve detecciones normalizadas con label/color/bbox
y track_id prefijado v-. No dibuja preview ni hace OCR.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.vehicles")

PADDLEX_URL = os.getenv("PADDLEX_URL", "http://paddlex:8080")
PADDLEX_PREDICT_PATH = os.getenv(
    "PADDLEX_PREDICT_PATH", "/vehicle-attribute-recognition"
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

_tracker = IoUTracker(IOU_THRESHOLD)


def reset_vehicle_tracker() -> None:
    """Reinicia el IoU tracker (llamar al abrir una foto nueva)."""
    global _tracker
    _tracker = IoUTracker(IOU_THRESHOLD)


def parse_attr_labels(
    labels: list[Any], scores: list[Any]
) -> tuple[Optional[str], Optional[str]]:
    """Extrae color y vehicle_type de labels tipo 'red(红色)', 'sedan(轿车)'.

    Solo conserva la parte latina antes del paréntesis; descarta CJK residual.
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
            color = raw
    return color, vtype


def parse_vehicle_boxes(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce respuesta vehicle_attribute_recognition → dicts crudos (sin track_id).

    Soporta shapes serving (`vehicles`) y predict local (`boxes`).
    Usado por infer full-frame y por InferenceSlicer (NMS-A).
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

    meta: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]

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

        color, vtype = parse_attr_labels(labels, scores)
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
                "entity_type": "vehicle",
            }
        )
    return meta


def finalize_vehicle_detections(
    meta: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Asigna track_id v-* / plate / frame_ts a dicts crudos de vehicles."""
    coords = [m["bbox"] for m in meta]
    track_ids = _tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    detections: list[dict[str, Any]] = []
    for tid, m in zip(track_ids, meta):
        detections.append(
            {
                "track_id": f"v-{tid}",
                "label": m["label"],
                "score": m["score"],
                "color": m.get("color"),
                "bbox": m["bbox"],
                "plate": None,
                "frame_ts": now,
                "entity_type": "vehicle",
            }
        )
    return detections


def normalize_vehicle_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce respuesta vehicle_attribute_recognition → dicts del adapter."""
    return finalize_vehicle_detections(parse_vehicle_boxes(data))


def infer_vehicles_tiled_sync(frame_hires) -> Optional[list[dict[str, Any]]]:
    """InferenceSlicer sync sobre hires; track_id post-slicer. None si todos fallan."""
    from detection.common.tiled_infer import infer_tiled_sync

    raw = infer_tiled_sync(
        frame_hires,
        base_url=PADDLEX_URL,
        predict_path=PADDLEX_PREDICT_PATH,
        normalize_response=parse_vehicle_boxes,
        capability="vehicles",
        timeout=HTTP_TIMEOUT,
        log=logger,
    )
    if raw is None:
        return None
    return finalize_vehicle_detections(raw)


def decode_paddlex_result_image(data: dict[str, Any]) -> Optional[bytes]:
    """Decodifica result.image Base64 del serving. None si falta o falla."""
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


async def infer_vehicles(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST JPEG a /vehicle-attribute-recognition. None si HTTP/errorCode falla."""
    from detection.common.paddlex_client import post_image_predict

    data = await post_image_predict(
        client,
        base_url=PADDLEX_URL,
        predict_path=PADDLEX_PREDICT_PATH,
        jpeg=jpeg,
        timeout=HTTP_TIMEOUT,
        log=logger,
        label="PaddleX vehicle",
        warn_on_error=True,
    )
    if data is None:
        return None
    return normalize_vehicle_result(data)
