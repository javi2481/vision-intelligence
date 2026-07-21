"""Overlay local de preview: bboxes + labels EN sobre el frame (sin CJK de PaddleX).

No usa result.image del serving. Colores por tipo de vehículo o fallback por track_id.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2

from detection.common.geometry import encode_jpeg

_PREVIEW_TYPE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "sedan": (255, 160, 0),
    "suv": (0, 140, 255),
    "van": (200, 0, 200),
    "truck": (0, 215, 255),
    "bus": (0, 0, 255),
    "mpv": (255, 0, 128),
    "pickup": (0, 200, 120),
    "hatchback": (180, 100, 0),
    "car": (255, 160, 0),
    "vehicle": (80, 80, 255),
}

_PREVIEW_FALLBACK_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 160, 0),
    (0, 140, 255),
    (200, 0, 200),
    (0, 215, 255),
    (0, 0, 255),
    (255, 0, 128),
    (0, 200, 120),
    (180, 100, 0),
)


def preview_label(det: dict[str, Any]) -> str:
    """Texto de etiqueta en inglés para el overlay (tipo + color)."""
    parts: list[str] = []
    label = det.get("label")
    color = det.get("color")
    if label:
        parts.append(str(label))
    if color:
        parts.append(str(color))
    return " ".join(parts) or "vehicle"


def preview_box_color(det: dict[str, Any]) -> tuple[int, int, int]:
    """Color BGR estable por tipo de vehículo (o track_id si falta tipo)."""
    label = str(det.get("label") or "").strip().lower()
    if label in _PREVIEW_TYPE_COLORS_BGR:
        return _PREVIEW_TYPE_COLORS_BGR[label]
    tid = str(det.get("track_id") or "0")
    digits = "".join(ch for ch in tid if ch.isdigit()) or "0"
    idx = int(digits) % len(_PREVIEW_FALLBACK_PALETTE_BGR)
    return _PREVIEW_FALLBACK_PALETTE_BGR[idx]


def draw_preview(frame, detections: list[dict[str, Any]]) -> Optional[bytes]:
    """Dibuja bboxes + labels EN y devuelve JPEG. None si el encode falla."""
    canvas = frame.copy()
    for det in detections or []:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1 = int(round(float(bbox[0])))
        y1 = int(round(float(bbox[1])))
        x2 = int(round(float(bbox[2])))
        y2 = int(round(float(bbox[3])))
        color = preview_box_color(det)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        text = preview_label(det)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        label_y1 = max(0, y1 - th - baseline - pad * 2)
        label_y2 = max(th + baseline + pad * 2, y1)
        label_x2 = min(canvas.shape[1] - 1, x1 + tw + pad * 2)
        cv2.rectangle(
            canvas, (x1, label_y1), (label_x2, label_y2), color, thickness=-1
        )
        cv2.putText(
            canvas,
            text,
            (x1 + pad, label_y2 - baseline - pad),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return encode_jpeg(canvas)
