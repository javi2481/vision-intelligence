"""Scaffold small_object_detection (experimental). GATE: ver README."""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.small_objects")

PADDLEX_SMALL_OBJECTS_URL = os.getenv(
    "PADDLEX_SMALL_OBJECTS_URL", "http://paddlex-small-objects:8091"
)
ENABLE_SMALL_OBJECTS = os.getenv(
    "ENABLE_SMALL_OBJECTS", "false"
).strip().lower() in ("1", "true", "yes")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
_tracker = IoUTracker(0.3)


def normalize_small_objects_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = data.get("result", data)
    boxes = result.get("boxes") if isinstance(result, dict) else []
    coords, meta = [], []
    for box in boxes if isinstance(boxes, list) else []:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(c) for c in coord[:4]]
        coords.append(bbox)
        meta.append(
            {
                "label": str(box.get("label") or "small_object"),
                "score": float(box.get("score") or 0.0),
                "bbox": bbox,
            }
        )
    tids = _tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"so-{tid}",
            "label": m["label"],
            "score": m["score"],
            "bbox": m["bbox"],
            "entity_type": "small_object",
            "frame_ts": now,
        }
        for tid, m in zip(tids, meta)
    ]


async def infer_small_objects(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    if not ENABLE_SMALL_OBJECTS:
        return None
    url = f"{PADDLEX_SMALL_OBJECTS_URL.rstrip('/')}/small-object-detection"
    try:
        resp = await client.post(
            url,
            json={"image": base64.b64encode(jpeg).decode("ascii")},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Small objects error (isolated): %s", exc)
        return None
    if not isinstance(data, dict) or data.get("errorCode") not in (None, 0, "0"):
        return None
    return normalize_small_objects_result(data)
