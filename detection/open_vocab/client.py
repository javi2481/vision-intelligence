"""Scaffold open-vocabulary detection (experimental). GATE: ver README."""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.open_vocab")

PADDLEX_OPEN_VOCAB_URL = os.getenv(
    "PADDLEX_OPEN_VOCAB_URL", "http://paddlex-open-vocab:8093"
)
ENABLE_OPEN_VOCAB = os.getenv("ENABLE_OPEN_VOCAB", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
OPEN_VOCAB_PROMPT = os.getenv("OPEN_VOCAB_PROMPT", "person,car,traffic sign")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
_tracker = IoUTracker(0.3)


def normalize_open_vocab_result(data: dict[str, Any]) -> list[dict[str, Any]]:
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
                "label": str(box.get("label") or "open"),
                "score": float(box.get("score") or 0.0),
                "bbox": bbox,
            }
        )
    tids = _tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"ov-{tid}",
            "label": m["label"],
            "score": m["score"],
            "bbox": m["bbox"],
            "entity_type": "open_vocab",
            "frame_ts": now,
        }
        for tid, m in zip(tids, meta)
    ]


async def infer_open_vocab(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    if not ENABLE_OPEN_VOCAB:
        return None
    url = f"{PADDLEX_OPEN_VOCAB_URL.rstrip('/')}/open-vocabulary-detection"
    try:
        resp = await client.post(
            url,
            json={
                "image": base64.b64encode(jpeg).decode("ascii"),
                "prompt": OPEN_VOCAB_PROMPT,
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Open-vocab error (isolated): %s", exc)
        return None
    if not isinstance(data, dict) or data.get("errorCode") not in (None, 0, "0"):
        return None
    return normalize_open_vocab_result(data)
