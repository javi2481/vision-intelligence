"""Scaffold anomaly_detection (experimental). GATE: ver README."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("detection.anomaly")

PADDLEX_ANOMALY_URL = os.getenv(
    "PADDLEX_ANOMALY_URL", "http://paddlex-anomaly:8092"
)
ENABLE_ANOMALY = os.getenv("ENABLE_ANOMALY", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))


def normalize_anomaly_result(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    result = data.get("result", data)
    if not isinstance(result, dict):
        return None
    score = float(result.get("score") or result.get("anomaly_score") or 0.0)
    label = str(result.get("label") or ("anomaly" if score > 0.5 else "normal"))
    now = datetime.now(timezone.utc).isoformat()
    return {
        "track_id": "anom-0",
        "label": label,
        "score": score,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "entity_type": "anomaly",
        "frame_ts": now,
    }


async def infer_anomaly(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[dict[str, Any]]:
    if not ENABLE_ANOMALY:
        return None
    from detection.common.paddlex_client import post_image_predict

    data = await post_image_predict(
        client,
        base_url=PADDLEX_ANOMALY_URL,
        predict_path="/anomaly-detection",
        jpeg=jpeg,
        timeout=HTTP_TIMEOUT,
        log=logger,
        label="Anomaly",
    )
    if data is None:
        return None
    return normalize_anomaly_result(data)
