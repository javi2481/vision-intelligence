"""Scaffold image_classification (profile experimental).

GATE: regla AMIS escrita + estimación RAM/latencia antes de ENABLE_SCENE_CLS=true.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("detection.scene_cls")

PADDLEX_SCENE_CLS_URL = os.getenv(
    "PADDLEX_SCENE_CLS_URL", "http://paddlex-scene-cls:8089"
)
PADDLEX_SCENE_CLS_PREDICT_PATH = os.getenv(
    "PADDLEX_SCENE_CLS_PREDICT_PATH", "/image-classification"
)
ENABLE_SCENE_CLS = os.getenv("ENABLE_SCENE_CLS", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))


def normalize_scene_cls_result(data: dict[str, Any]) -> Optional[dict[str, Any]]:
    result = data.get("result", data)
    if not isinstance(result, dict):
        if isinstance(result, list) and result and isinstance(result[0], dict):
            result = result[0]
        else:
            return None
    label = result.get("label") or result.get("class_name")
    scores = result.get("scores") or result.get("cls_scores") or []
    labels = result.get("labels") or result.get("class_names") or []
    # Serving 3.7: InferResult.categories = [{id, name, score}, ...]
    categories = result.get("categories") or []
    if not label and isinstance(categories, list) and categories:
        top = categories[0] if isinstance(categories[0], dict) else None
        if top:
            label = top.get("name") or top.get("label")
            if not scores and top.get("score") is not None:
                scores = [top["score"]]
    if not label and labels:
        label = labels[0]
    score = float(scores[0]) if scores else float(result.get("score") or 0.0)
    if not label:
        return None
    now = datetime.now(timezone.utc).isoformat()
    return {
        "track_id": "scls-0",
        "label": str(label),
        "score": score,
        "bbox": [0.0, 0.0, 1.0, 1.0],
        "entity_type": "scene_cls",
        "frame_ts": now,
    }


async def infer_scene_cls(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[dict[str, Any]]:
    if not ENABLE_SCENE_CLS:
        return None
    url = f"{PADDLEX_SCENE_CLS_URL.rstrip('/')}{PADDLEX_SCENE_CLS_PREDICT_PATH}"
    try:
        resp = await client.post(
            url,
            json={
                "image": base64.b64encode(jpeg).decode("ascii"),
                "visualize": False,
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Scene cls error (isolated): %s", exc)
        return None
    if not isinstance(data, dict) or data.get("errorCode") not in (None, 0, "0"):
        return None
    return normalize_scene_cls_result(data)
