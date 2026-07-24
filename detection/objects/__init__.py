"""Detección general COCO (incluye person como clase, no como pipeline aparte)."""

from detection.objects.client import (
    VEHICLE_COCO_LABELS,
    attach_object_track_ids,
    infer_objects,
    infer_objects_tiled_sync,
    merge_coco_detections,
    normalize_object_detection_result,
    reset_object_tracker,
)

__all__ = [
    "VEHICLE_COCO_LABELS",
    "attach_object_track_ids",
    "infer_objects",
    "infer_objects_tiled_sync",
    "merge_coco_detections",
    "normalize_object_detection_result",
    "reset_object_tracker",
]
