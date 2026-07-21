"""OCR de escena / carteles (no solo patentes)."""

from detection.text.client import (
    ENABLE_SCENE_OCR,
    infer_scene_ocr,
    normalize_scene_ocr_result,
)

__all__ = [
    "ENABLE_SCENE_OCR",
    "infer_scene_ocr",
    "normalize_scene_ocr_result",
]
