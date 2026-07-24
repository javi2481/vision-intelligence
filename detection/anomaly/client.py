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
PADDLEX_ANOMALY_PREDICT_PATH = os.getenv(
    "PADDLEX_ANOMALY_PREDICT_PATH", "/image-anomaly-detection"
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
    score_raw = result.get("score")
    if score_raw is None:
        score_raw = result.get("anomaly_score")
    if score_raw is None and isinstance(result.get("labelMap"), list):
        # Serving PaddleX 3.7: labelMap 0=normal, 255=anomaly (pixel mask).
        label_map = result["labelMap"]
        n = len(label_map)
        if n == 0:
            return None
        anom = sum(1 for v in label_map if int(v) != 0)
        score_raw = anom / n
    score = float(score_raw or 0.0)
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
        predict_path=PADDLEX_ANOMALY_PREDICT_PATH,
        jpeg=jpeg,
        timeout=HTTP_TIMEOUT,
        log=logger,
        label="Anomaly",
    )
    if data is None:
        return None
    return normalize_anomaly_result(data)
