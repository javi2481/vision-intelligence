"""Instance segmentation (experimental)."""

from detection.instances.client import (
    ENABLE_INSTANCE_SEG,
    infer_instances,
    normalize_instances_result,
)

__all__ = ["ENABLE_INSTANCE_SEG", "infer_instances", "normalize_instances_result"]
