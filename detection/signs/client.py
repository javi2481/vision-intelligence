"""Señales de tránsito vía object_detection dedicado (PaddleX :8088).

Opcional (ENABLE_SIGNS). Por default filtra labels de señales COCO / custom.
Con fine-tune, apuntar el servicio a pesos propios. Caída aislada.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.signs")

PADDLEX_SIGNS_URL = os.getenv("PADDLEX_SIGNS_URL", "http://paddlex-signs:8088")
PADDLEX_SIGNS_PREDICT_PATH = os.getenv(
    "PADDLEX_SIGNS_PREDICT_PATH", "/object-detection"
)
ENABLE_SIGNS = os.getenv("ENABLE_SIGNS", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

# COCO + nombres típicos de fine-tune
DEFAULT_SIGN_LABELS = {
    "traffic light",
    "stop sign",
    "parking meter",
    "fire hydrant",
    "traffic_sign",
    "traffic sign",
    "sign",
    "speed_limit",
    "yield",
    "crosswalk_sign",
}

_raw = os.getenv("SIGNS_LABELS", "")
SIGN_LABELS = (
    {s.strip().lower() for s in _raw.split(",") if s.strip()}
    if _raw.strip()
    else DEFAULT_SIGN_LABELS
)

_signs_tracker = IoUTracker(IOU_THRESHOLD)


def reset_signs_tracker() -> None:
    global _signs_tracker
    _signs_tracker = IoUTracker(IOU_THRESHOLD)


def normalize_signs_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce object_detection → dets sign filtradas."""
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

    coords: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        label = str(box.get("label") or box.get("cls_name") or "").strip().lower()
        if SIGN_LABELS and label not in SIGN_LABELS:
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
        score = float(box.get("score") or box.get("det_score") or 0.0)
        coords.append(bbox)
        meta.append({"label": label or "sign", "score": score, "bbox": bbox})

    track_ids = _signs_tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"s-{tid}",
            "label": m["label"],
            "score": m["score"],
            "bbox": m["bbox"],
            "entity_type": "sign",
            "frame_ts": now,
        }
        for tid, m in zip(track_ids, meta)
    ]


async def infer_signs(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST JPEG a object-detection (signs). None ante fallo."""
    if not ENABLE_SIGNS:
        return None
    url = f"{PADDLEX_SIGNS_URL.rstrip('/')}{PADDLEX_SIGNS_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Signs infer error (isolated): %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.debug("Signs error: %s", data.get("errorMsg"))
        return None
    return normalize_signs_result(data)
