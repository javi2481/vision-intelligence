"""Small object detection (experimental)."""

from detection.small_objects.client import (
    ENABLE_SMALL_OBJECTS,
    infer_small_objects,
    normalize_small_objects_result,
)

__all__ = [
    "ENABLE_SMALL_OBJECTS",
    "infer_small_objects",
    "normalize_small_objects_result",
]
