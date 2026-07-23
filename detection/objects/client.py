"""Cliente HTTP al pipeline object_detection COCO (PaddleX :8082).

Cubre ~80 clases (person, dog, bottle, …). Las clases vehículo COCO se
dedupean contra el pipeline de vehicles (más rico en color/plate).
Caída aislada: nunca degrada el pipeline primario de vehículos.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker, iou

logger = logging.getLogger("detection.objects")

PADDLEX_OBJECTS_URL = os.getenv("PADDLEX_OBJECTS_URL", "http://paddlex-objects:8082")
PADDLEX_OBJECTS_PREDICT_PATH = os.getenv(
    "PADDLEX_OBJECTS_PREDICT_PATH", "/object-detection"
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

# Labels COCO ya cubiertos por vehicle_attribute_recognition (color/plate).
VEHICLE_COCO_LABELS = {"car", "truck", "bus", "motorcycle", "bicycle"}

_object_tracker = IoUTracker(IOU_THRESHOLD)


def reset_object_tracker() -> None:
    """Reinicia el IoU tracker de objetos (llamar al abrir una foto nueva)."""
    global _object_tracker
    _object_tracker = IoUTracker(IOU_THRESHOLD)


def normalize_object_detection_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce respuesta object_detection → dicts crudos (sin track_id aún).

    Acepta el schema legacy (`boxes` + `label`/`coordinate`) y el de PaddleX
    actual (`detectedObjects` + `categoryName`/`bbox`).
    """
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = result.get("detectedObjects") or result.get("boxes") or []
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        for item in result:
            if not isinstance(item, dict):
                continue
            if "boxes" in item:
                boxes.extend(item.get("boxes") or [])
            elif "detectedObjects" in item:
                boxes.extend(item.get("detectedObjects") or [])

    detections: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
        label = (
            box.get("label")
            or box.get("categoryName")
            or box.get("cls_name")
            or ""
        )
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


def merge_coco_detections(
    vehicle_dets: list[dict[str, Any]],
    object_dets: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Descarta objetos COCO de clase vehículo ya cubiertos por el pipeline vehicle.

    Labels no-vehículo (person, dog, …) se conservan siempre.
    """
    if not object_dets:
        return []
    vehicle_boxes = [v["bbox"] for v in vehicle_dets if v.get("bbox")]
    kept: list[dict[str, Any]] = []
    for det in object_dets:
        label = str(det.get("label") or "").strip().lower()
        bbox = det.get("bbox")
        if label in VEHICLE_COCO_LABELS and bbox and vehicle_boxes:
            if any(iou(bbox, vb) > iou_threshold for vb in vehicle_boxes):
                continue
        kept.append(det)
    return kept


def attach_object_track_ids(
    object_raw: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Asigna track_id o-* y frame_ts a detecciones COCO crudas."""
    object_boxes = [d["bbox"] for d in object_raw]
    object_track_ids = _object_tracker.assign(object_boxes)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"o-{tid}",
            "label": d["label"],
            "score": d["score"],
            "bbox": d["bbox"],
            "entity_type": "object",
            "frame_ts": now,
        }
        for tid, d in zip(object_track_ids, object_raw)
    ]


async def infer_objects(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST JPEG a object-detection. None ante fallo (aislado, no degrada)."""
    url = f"{PADDLEX_OBJECTS_URL.rstrip('/')}{PADDLEX_OBJECTS_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Object detection infer error (isolated): %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.debug("Object detection error: %s", data.get("errorMsg"))
        return None
    return normalize_object_detection_result(data)
