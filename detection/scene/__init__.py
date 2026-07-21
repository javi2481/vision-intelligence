"""Escena vial vía semantic_segmentation (Cityscapes + lane + bdd_marks)."""

from detection.scene.client import (
    ENABLE_SCENE_SEG,
    PADDLEX_SCENE_URL,
    build_crosswalk_from_ratios,
    build_infra,
    build_lanes_from_ratios,
    class_ratios_from_label_map,
    infer_scene,
    infer_scene_type,
    normalize_scene_result,
)

__all__ = [
    "ENABLE_SCENE_SEG",
    "PADDLEX_SCENE_URL",
    "build_crosswalk_from_ratios",
    "build_infra",
    "build_lanes_from_ratios",
    "class_ratios_from_label_map",
    "infer_scene",
    "infer_scene_type",
    "normalize_scene_result",
]
