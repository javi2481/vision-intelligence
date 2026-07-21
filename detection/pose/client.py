"""Cliente HTTP al pipeline human_keypoint_detection (PaddleX :8086).

Opcional (ENABLE_POSE). Emite entity_type=pose con keypoints.
Caída aislada: no degrada el bridge.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import IoUTracker

logger = logging.getLogger("detection.pose")

PADDLEX_POSE_URL = os.getenv("PADDLEX_POSE_URL", "http://paddlex-pose:8086")
PADDLEX_POSE_PREDICT_PATH = os.getenv(
    "PADDLEX_POSE_PREDICT_PATH", "/human-keypoint-detection"
)
ENABLE_POSE = os.getenv("ENABLE_POSE", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

_pose_tracker = IoUTracker(IOU_THRESHOLD)


def reset_pose_tracker() -> None:
    """Reinicia tracker de poses al abrir una foto nueva."""
    global _pose_tracker
    _pose_tracker = IoUTracker(IOU_THRESHOLD)


def _bbox_from_keypoints(kps: list[Any]) -> Optional[list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for kp in kps:
        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
            xs.append(float(kp[0]))
            ys.append(float(kp[1]))
        elif isinstance(kp, dict):
            if "x" in kp and "y" in kp:
                xs.append(float(kp["x"]))
                ys.append(float(kp["y"]))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_pose_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce human_keypoint_detection → dets pose con track k-*."""
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = (
            result.get("boxes")
            or result.get("poses")
            or result.get("keypoints")
            or []
        )
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        boxes = [x for x in result if isinstance(x, dict)]

    coords: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        kps = box.get("keypoints") or box.get("kpts") or box.get("keypoint") or []
        if not coord or len(coord) < 4:
            coord = _bbox_from_keypoints(kps if isinstance(kps, list) else [])
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]
        score = float(box.get("score") or box.get("det_score") or 0.0)
        coords.append(bbox)
        meta.append(
            {
                "score": score,
                "bbox": bbox,
                "keypoints": kps if isinstance(kps, list) else [],
            }
        )

    track_ids = _pose_tracker.assign(coords)
    now = datetime.now(timezone.utc).isoformat()
    return [
        {
            "track_id": f"k-{tid}",
            "label": "person_pose",
            "score": m["score"],
            "bbox": m["bbox"],
            "entity_type": "pose",
            "keypoints": m["keypoints"],
            "frame_ts": now,
        }
        for tid, m in zip(track_ids, meta)
    ]


async def infer_pose(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST JPEG a human-keypoint-detection. None ante fallo."""
    if not ENABLE_POSE:
        return None
    url = f"{PADDLEX_POSE_URL.rstrip('/')}{PADDLEX_POSE_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Pose infer error (isolated): %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.debug("Pose error: %s", data.get("errorMsg"))
        return None
    return normalize_pose_result(data)
