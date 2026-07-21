"""OCR de escena / carteles (reusa paddlex-ocr :8081).

Opcional (ENABLE_SCENE_OCR). Distinto de plates/: no filtra solo patentes;
devuelve líneas de texto con score. Caída aislada.
"""

from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("detection.text")

PADDLEX_OCR_URL = os.getenv("PADDLEX_OCR_URL", "http://paddlex-ocr:8081")
ENABLE_SCENE_OCR = os.getenv("ENABLE_SCENE_OCR", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
SCENE_OCR_MIN_SCORE = float(os.getenv("SCENE_OCR_MIN_SCORE", "0.5"))
SCENE_OCR_MAX_LINES = max(1, int(os.getenv("SCENE_OCR_MAX_LINES", "20")))
OCR_HTTP_TIMEOUT = float(os.getenv("OCR_HTTP_TIMEOUT", "5"))


def normalize_scene_ocr_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce respuesta OCR general → dets entity_type=text."""
    result = data.get("result", data) if isinstance(data, dict) else {}
    if not isinstance(result, dict):
        return []

    lines: list[tuple[str, float, Optional[list[float]]]] = []

    ocr_results = result.get("ocrResults") or []
    if isinstance(ocr_results, list) and ocr_results:
        first = ocr_results[0] if isinstance(ocr_results[0], dict) else {}
        pruned = first.get("prunedResult") if isinstance(first, dict) else {}
        if isinstance(pruned, dict):
            texts = pruned.get("rec_texts") or []
            scores = pruned.get("rec_scores") or []
            polys = pruned.get("dt_polys") or pruned.get("rec_polys") or []
            for i, text in enumerate(texts):
                score = float(scores[i]) if i < len(scores) else 0.0
                bbox = None
                if i < len(polys) and isinstance(polys[i], (list, tuple)):
                    pts = polys[i]
                    try:
                        xs = [float(p[0]) for p in pts]
                        ys = [float(p[1]) for p in pts]
                        bbox = [min(xs), min(ys), max(xs), max(ys)]
                    except (TypeError, ValueError, IndexError):
                        bbox = None
                lines.append((str(text), score, bbox))

    # shape alternativo: texts / scores en raíz
    if not lines:
        texts = result.get("rec_texts") or result.get("texts") or []
        scores = result.get("rec_scores") or result.get("scores") or []
        for i, text in enumerate(texts):
            score = float(scores[i]) if i < len(scores) else 0.0
            lines.append((str(text), score, None))

    lines = sorted(lines, key=lambda t: t[1], reverse=True)
    lines = [
        (t, s, b)
        for t, s, b in lines
        if s >= SCENE_OCR_MIN_SCORE and str(t).strip()
    ][:SCENE_OCR_MAX_LINES]

    now = datetime.now(timezone.utc).isoformat()
    dets: list[dict[str, Any]] = []
    for i, (text, score, bbox) in enumerate(lines):
        dets.append(
            {
                "track_id": f"t-{i}",
                "label": "text",
                "score": score,
                "bbox": bbox or [0.0, 0.0, 1.0, 1.0],
                "entity_type": "text",
                "text": text.strip(),
                "frame_ts": now,
            }
        )
    return dets


async def infer_scene_ocr(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST frame completo a /ocr. None ante fallo."""
    if not ENABLE_SCENE_OCR:
        return None
    url = f"{PADDLEX_OCR_URL.rstrip('/')}/ocr"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url,
            json={"file": b64, "fileType": 1},
            timeout=OCR_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Scene OCR infer error (isolated): %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    return normalize_scene_ocr_result(data)
