"""OCR de patentes (opcional) sobre crops de vehículos detectados."""

from detection.plates.client import (
    ENABLE_PLATE_OCR,
    OCR_MIN_SCORE,
    OCR_TOPK,
    PADDLEX_OCR_URL,
    crop_bbox,
    enrich_vehicles_with_plates,
    infer_plate_ocr,
    parse_plate,
    plate_parse_stats,
    reset_plate_parse_stats,
)

__all__ = [
    "ENABLE_PLATE_OCR",
    "OCR_MIN_SCORE",
    "OCR_TOPK",
    "PADDLEX_OCR_URL",
    "crop_bbox",
    "enrich_vehicles_with_plates",
    "infer_plate_ocr",
    "parse_plate",
    "plate_parse_stats",
    "reset_plate_parse_stats",
]
