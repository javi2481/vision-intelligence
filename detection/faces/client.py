"""Cliente HTTP al pipeline face_detection (PaddleX :8083).

Opcional (ENABLE_FACE_DETECTION). Caída aislada: no degrada el bridge.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.faces")

PADDLEX_FACES_URL = os.getenv("PADDLEX_FACES_URL", "http://paddlex-faces:8083")
# Pipeline custom (object_detection + face model) → endpoint OD.
PADDLEX_FACES_PREDICT_PATH = os.getenv(
    "PADDLEX_FACES_PREDICT_PATH", "/object-detection"
)
ENABLE_FACE_DETECTION = os.getenv(
    "ENABLE_FACE_DETECTION", "false"
).strip().lower() in ("1", "true", "yes")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

_face_tracker = IoUTracker(IOU_THRESHOLD)


def reset_face_tracker() -> None:
    """Reinicia el IoU tracker de rostros (llamar al abrir una foto nueva)."""
    global _face_tracker
    _face_tracker = IoUTracker(IOU_THRESHOLD)


def normalize_face_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce respuesta face_detection → dicts con track_id f-*."""
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = (
            result.get("detectedObjects")
            or result.get("boxes")
            or result.get("faces")
            or []
        )
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and "boxes" in item:
                boxes.extend(item.get("boxes") or [])
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
        score = float(box.get("score") or box.get("det_score") or 0.0)
        coords.append(bbox)
        meta.append({"score": score, "bbox": bbox})

    track_ids = _face_tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"f-{tid}",
            "label": "face",
            "score": m["score"],
            "bbox": m["bbox"],
            "entity_type": "face",
            "frame_ts": now,
        }
        for tid, m in zip(track_ids, meta)
    ]


async def infer_faces(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST JPEG a face-detection. None ante fallo (aislado)."""
    if not ENABLE_FACE_DETECTION:
        return None
    from detection.common.paddlex_client import post_image_predict

    data = await post_image_predict(
        client,
        base_url=PADDLEX_FACES_URL,
        predict_path=PADDLEX_FACES_PREDICT_PATH,
        jpeg=jpeg,
        timeout=HTTP_TIMEOUT,
        log=logger,
        label="Face detection",
    )
    if data is None:
        return None
    return normalize_face_result(data)
