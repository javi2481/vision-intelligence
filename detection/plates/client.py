"""Cliente OCR de patente (PaddleX paddlex-ocr :8081).

Opcional (ENABLE_PLATE_OCR). Crop del bbox del vehículo → POST /ocr →
mejor match regex 5-8 alfanuméricos. Caída aislada: solo deja plate=None.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from typing import Any, Optional

import httpx

logger = logging.getLogger("detection.plates")

PADDLEX_OCR_URL = os.getenv("PADDLEX_OCR_URL", "http://paddlex-ocr:8081")
ENABLE_PLATE_OCR = os.getenv("ENABLE_PLATE_OCR", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
OCR_MIN_SCORE = float(os.getenv("OCR_MIN_SCORE", "0.7"))
OCR_TOPK = max(1, int(os.getenv("OCR_TOPK", "3")))
OCR_HTTP_TIMEOUT = float(os.getenv("OCR_HTTP_TIMEOUT", "5"))
PLATE_REGEX = re.compile(r"^[A-Z0-9]{5,8}$")
_OCR_MIN_CROP_PX = 8

# parse_plate instrumentation (process-local; reset between tests/runs).
_plate_parse_stats: dict[str, int] = {
    "total": 0,
    "rejected_regex": 0,
    "accepted": 0,
}


def reset_plate_parse_stats() -> None:
    """Zero parse_plate counters (tests / measurement runs)."""
    _plate_parse_stats["total"] = 0
    _plate_parse_stats["rejected_regex"] = 0
    _plate_parse_stats["accepted"] = 0


def plate_parse_stats() -> dict[str, int]:
    """Snapshot of parse_plate counters: total / rejected_regex / accepted."""
    return dict(_plate_parse_stats)


def crop_bbox(frame, bbox: list[float]) -> Optional[Any]:
    """Recorta frame al bbox clippeado. None si el crop es degenerado."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), w))
    y1 = max(0, min(int(round(y1)), h))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if (x2 - x1) < _OCR_MIN_CROP_PX or (y2 - y1) < _OCR_MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def parse_plate(
    rec_texts: list[Any], rec_scores: list[Any]
) -> Optional[dict[str, Any]]:
    """Normaliza textos OCR y devuelve el match de patente de mayor score.

    Counters (see plate_parse_stats): total candidates, rejected_regex, accepted
    (candidates that match PLATE_REGEX).
    """
    best: Optional[dict[str, Any]] = None
    for text, score in zip(rec_texts or [], rec_scores or []):
        _plate_parse_stats["total"] += 1
        normalized = re.sub(r"[^A-Z0-9]", "", str(text).upper())
        if not PLATE_REGEX.match(normalized):
            _plate_parse_stats["rejected_regex"] += 1
            continue
        _plate_parse_stats["accepted"] += 1
        score_f = float(score or 0.0)
        if best is None or score_f > best["score"]:
            best = {"text": normalized, "score": score_f}
    return best


async def infer_plate_ocr(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[dict[str, Any]]:
    """POST crop JPEG a /ocr. None ante fallo o sin match de patente."""
    url = f"{PADDLEX_OCR_URL.rstrip('/')}/ocr"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"file": b64, "fileType": 1}, timeout=OCR_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("OCR infer error (isolated): %s", exc)
        return None

    try:
        result = data.get("result", data) if isinstance(data, dict) else {}
        ocr_results = result.get("ocrResults") or []
        if not ocr_results:
            return None
        pruned = ocr_results[0].get("prunedResult") or {}
        rec_texts = pruned.get("rec_texts") or []
        rec_scores = pruned.get("rec_scores") or []
    except Exception as exc:
        logger.debug("OCR result parse error: %s", exc)
        return None
    return parse_plate(rec_texts, rec_scores)


async def enrich_vehicles_with_plates(
    client: httpx.AsyncClient,
    frame_hires,
    vehicle_detections: list[dict[str, Any]],
    encode_jpeg_fn,
) -> None:
    """Si ENABLE_PLATE_OCR, completa d['plate'] en top-K vehículos elegibles.

    Mutates vehicle_detections in-place. encode_jpeg_fn: callable(frame) -> bytes|None.
    """
    if not ENABLE_PLATE_OCR or not vehicle_detections:
        return
    eligible = sorted(
        (d for d in vehicle_detections if d.get("score", 0.0) > OCR_MIN_SCORE),
        key=lambda d: d["score"],
        reverse=True,
    )[:OCR_TOPK]
    for d in eligible:
        crop = crop_bbox(frame_hires, d["bbox"])
        if crop is None:
            continue
        crop_jpeg = encode_jpeg_fn(crop)
        if crop_jpeg is None:
            continue
        plate = await infer_plate_ocr(client, crop_jpeg)
        if plate:
            d["plate"] = plate
