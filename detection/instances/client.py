"""Scaffold instance_segmentation (experimental). GATE: ver README."""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.instances")

PADDLEX_INSTANCES_URL = os.getenv(
    "PADDLEX_INSTANCES_URL", "http://paddlex-instances:8090"
)
ENABLE_INSTANCE_SEG = os.getenv(
    "ENABLE_INSTANCE_SEG", "false"
).strip().lower() in ("1", "true", "yes")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
_tracker = IoUTracker(0.3)


def normalize_instances_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = data.get("result", data)
    boxes = []
    if isinstance(result, dict):
        boxes = result.get("boxes") or result.get("instances") or []
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
                "label": str(box.get("label") or "instance"),
                "score": float(box.get("score") or 0.0),
                "bbox": bbox,
            }
        )
    tids = _tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"i-{tid}",
            "label": m["label"],
            "score": m["score"],
            "bbox": m["bbox"],
            "entity_type": "instance",
            "frame_ts": now,
        }
        for tid, m in zip(tids, meta)
    ]


async def infer_instances(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    if not ENABLE_INSTANCE_SEG:
        return None
    url = f"{PADDLEX_INSTANCES_URL.rstrip('/')}/instance-segmentation"
    try:
        resp = await client.post(
            url,
            json={"image": base64.b64encode(jpeg).decode("ascii")},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Instances error (isolated): %s", exc)
        return None
    if not isinstance(data, dict) or data.get("errorCode") not in (None, 0, "0"):
        return None
    return normalize_instances_result(data)
