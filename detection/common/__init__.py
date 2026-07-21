"""Utilidades compartidas: tracking IoU, geometría de frame, preview overlay."""

from detection.common.geometry import (
    BRIDGE_MAX_WIDTH,
    JPEG_QUALITY,
    encode_jpeg,
    maybe_resize_for_infer,
    scale_detections,
)
from detection.common.preview import draw_preview, preview_box_color, preview_label
from detection.common.tracking import IoUTracker, iou

__all__ = [
    "BRIDGE_MAX_WIDTH",
    "JPEG_QUALITY",
    "IoUTracker",
    "draw_preview",
    "encode_jpeg",
    "iou",
    "maybe_resize_for_infer",
    "preview_box_color",
    "preview_label",
    "scale_detections",
]
