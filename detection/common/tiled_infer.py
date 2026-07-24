"""InferenceSlicer sync core (NMS capa A) para vehicles/objects.

Coordena tiles sobre frame_hires; el slicer deja cajas en coords del frame
completo via move_detections. No asigna track_id (post-slicer en el caller).
asyncio.to_thread vive solo en bridge/main.py — este módulo es sync puro.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

import httpx
import numpy as np
import supervision as sv

from detection.common.geometry import encode_jpeg
from detection.common.paddlex_client import env_flag, post_image_predict_sync

logger = logging.getLogger("detection.tiled_infer")

ENABLE_INFER_TILING = env_flag("ENABLE_INFER_TILING", "false")
# Defaults: INFER_SLICE_WH from PR1 measure; overlap/workers/iou = supervision 0.28 API defaults.
# https://supervision.roboflow.com/0.28.0/detection/tools/inference_slicer/
INFER_SLICE_WH = int(os.getenv("INFER_SLICE_WH", "640"))
INFER_OVERLAP_WH = int(os.getenv("INFER_OVERLAP_WH", "100"))
INFER_TILE_THREAD_WORKERS = int(os.getenv("INFER_TILE_THREAD_WORKERS", "1"))

# Mapas label→id **intra-capacidad** (NMS-A). No reutilizar en NMS-B
# (ver detection.common.nms_cross_cap.class_id_for_cross_cap_nms).
_TILE_CLASS_IDS: dict[str, dict[str, int]] = {}


def class_id_for_tile_nms(label: str, *, capability: str) -> int:
    """Id estable por label dentro de una capacidad (solo NMS del slicer / capa A)."""
    key = (label or "").strip().lower() or "_unknown"
    bucket = _TILE_CLASS_IDS.setdefault(capability, {})
    if key not in bucket:
        bucket[key] = len(bucket)
    return bucket[key]


def _wh_tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return (int(value[0]), int(value[1]))
    v = int(value)
    return (v, v)


def vi_raw_to_detections(
    raw: list[dict[str, Any]], *, capability: str
) -> sv.Detections:
    """Dicts VI crudos (label/score/bbox[+color]) → Detections con class_id NMS-A."""
    if not raw:
        return sv.Detections.empty()
    xyxy = np.array([d["bbox"] for d in raw], dtype=np.float32)
    confidence = np.array([float(d.get("score") or 0.0) for d in raw], dtype=np.float32)
    class_id = np.array(
        [class_id_for_tile_nms(str(d.get("label") or ""), capability=capability)
         for d in raw],
        dtype=np.int32,
    )
    data: dict[str, Any] = {
        "label": np.array([str(d.get("label") or "") for d in raw], dtype=object),
    }
    if capability == "vehicles":
        data["color"] = np.array(
            [d.get("color") for d in raw], dtype=object
        )
    return sv.Detections(
        xyxy=xyxy, confidence=confidence, class_id=class_id, data=data
    )


def detections_to_vi_raw(
    detections: sv.Detections, *, capability: str
) -> list[dict[str, Any]]:
    """Detections post-slicer → dicts VI sin track_id (coords hires)."""
    out: list[dict[str, Any]] = []
    n = len(detections)
    labels = detections.data.get("label") if detections.data else None
    colors = detections.data.get("color") if detections.data else None
    for i in range(n):
        label = str(labels[i]) if labels is not None else ""
        score = (
            float(detections.confidence[i])
            if detections.confidence is not None
            else 0.0
        )
        row: dict[str, Any] = {
            "label": label,
            "score": score,
            "bbox": [float(x) for x in detections.xyxy[i]],
            "entity_type": "vehicle" if capability == "vehicles" else "object",
        }
        if capability == "vehicles":
            row["color"] = colors[i] if colors is not None else None
        out.append(row)
    return out


NormalizeFn = Callable[[dict[str, Any]], list[dict[str, Any]]]


def infer_tiled_sync(
    frame_hires: Any,
    *,
    base_url: str,
    predict_path: str,
    normalize_response: NormalizeFn,
    capability: str,
    slice_wh: int | tuple[int, int] | None = None,
    overlap_wh: int | tuple[int, int] | None = None,
    thread_workers: int | None = None,
    timeout: float = 30.0,
    log: Optional[logging.Logger] = None,
) -> Optional[list[dict[str, Any]]]:
    """Corre InferenceSlicer sobre frame_hires; None si todos los tiles fallan HTTP.

    Returns list (posiblemente vacía) de dicts sin track_id, cajas en coords hires.
    """
    log = log or logger
    sw = _wh_tuple(slice_wh if slice_wh is not None else INFER_SLICE_WH)
    ow = _wh_tuple(overlap_wh if overlap_wh is not None else INFER_OVERLAP_WH)
    workers = (
        INFER_TILE_THREAD_WORKERS if thread_workers is None else int(thread_workers)
    )
    if ow[0] >= sw[0] or ow[1] >= sw[1]:
        raise ValueError(
            f"INFER_OVERLAP_WH {ow} must be strictly < INFER_SLICE_WH {sw}"
        )

    successes = 0
    failures = 0

    with httpx.Client(timeout=timeout) as client:

        def callback(image_slice: np.ndarray) -> sv.Detections:
            nonlocal successes, failures
            jpeg = encode_jpeg(image_slice)
            if jpeg is None:
                failures += 1
                return sv.Detections.empty()
            data = post_image_predict_sync(
                client,
                base_url=base_url,
                predict_path=predict_path,
                jpeg=jpeg,
                timeout=timeout,
                log=log,
                label=f"tiled-{capability}",
                warn_on_error=False,
            )
            if data is None:
                failures += 1
                return sv.Detections.empty()
            successes += 1
            raw = normalize_response(data)
            return vi_raw_to_detections(raw, capability=capability)

        # Defaults iou_threshold=0.5 / overlap_metric=IOU / NON_MAX_SUPPRESSION
        # match supervision 0.28 InferenceSlicer (NMS capa A). After each tile,
        # slicer calls move_detections(offset) → boxes in full-frame coords.
        slicer = sv.InferenceSlicer(
            callback=callback,
            slice_wh=sw,
            overlap_wh=ow,
            overlap_filter=sv.OverlapFilter.NON_MAX_SUPPRESSION,
            overlap_metric=sv.OverlapMetric.IOU,
            thread_workers=workers,
        )
        merged = slicer(frame_hires)

    if successes == 0 and failures > 0:
        log.warning(
            "tiled-%s: all %d tile POSTs failed", capability, failures
        )
        return None
    return detections_to_vi_raw(merged, capability=capability)
