"""Clasificación global de escena (experimental)."""

from detection.scene_cls.client import (
    ENABLE_SCENE_CLS,
    infer_scene_cls,
    normalize_scene_cls_result,
)

__all__ = ["ENABLE_SCENE_CLS", "infer_scene_cls", "normalize_scene_cls_result"]
