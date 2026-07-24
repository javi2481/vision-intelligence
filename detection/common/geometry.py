"""Geometría de frame: encode JPEG, resize de inferencia, reescalado de bboxes.

Opera sobre frames OpenCV (BGR). No llama a PaddleX ni al adapter.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import cv2

JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))
# Ancho máximo del JPEG de caps no tileadas; frame_hires para OCR/crop/preview.
# Con ENABLE_INFER_TILING, vehicles/objects usan INFER_SLICE_WH sobre hires.
BRIDGE_MAX_WIDTH = int(os.getenv("BRIDGE_MAX_WIDTH", "1920"))


def encode_jpeg(frame, quality: Optional[int] = None) -> Optional[bytes]:
    """Codifica un frame BGR a JPEG. None si imencode falla."""
    q = JPEG_QUALITY if quality is None else quality
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return None
    return buf.tobytes()


def maybe_resize_for_infer(frame_hires) -> tuple[Any, float, float]:
    """Deriva frame_infer; downscale solo si el ancho supera BRIDGE_MAX_WIDTH.

    Retorna (frame_infer, scale_x, scale_y) donde scale_* multiplica
    coordenadas infer → hires. Pass-through (mismo objeto, 1.0/1.0) si no hace falta.
    """
    h, w = frame_hires.shape[:2]
    if w <= BRIDGE_MAX_WIDTH:
        return frame_hires, 1.0, 1.0
    new_w = BRIDGE_MAX_WIDTH
    new_h = max(1, round(h * new_w / w))
    frame_infer = cv2.resize(
        frame_hires, (new_w, new_h), interpolation=cv2.INTER_AREA
    )
    return frame_infer, w / new_w, h / new_h


def scale_detections(
    dets: list[dict[str, Any]], scale_x: float, scale_y: float
) -> list[dict[str, Any]]:
    """Escala bbox de coords frame_infer → frame_hires in-place. Pass-through si 1.0."""
    if scale_x == 1.0 and scale_y == 1.0:
        return dets
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        d["bbox"] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
    return dets
