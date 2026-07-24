"""Shared pack registry for PaddleX eval harness (Core + Extended + Experimental).

Layers:
  - core: local accuracy gate (default --packs core)
  - extended: opt-in via --packs all
  - experimental: opt-in via --packs all; never required for Core gate
"""

from __future__ import annotations

from typing import Any

SEED = 51
DEFAULT_MAX_SAMPLES = 20
FLAT_PREFIX = {
    "objects": "fo_objects_",
    "vehicles": "fo_vehicles_",
    "ocr_plates": "fo_ocr_plates_",
    "signs": "fo_signs_",
    "faces": "fo_faces_",
    "pose": "fo_pose_",
    "pedestrians": "fo_pedestrians_",
    "scene": "fo_scene_",
    "ocr_text": "fo_ocr_text_",
    "instances": "fo_instances_",
    "small_objects": "fo_small_objects_",
    "open_vocab": "fo_open_vocab_",
    "scene_cls": "fo_scene_cls_",
    "anomaly": "fo_anomaly_",
    "face_id": "fo_face_id_",
}

# COCO category names used when filtering the zoo.
VEHICLE_COCO_LABELS = frozenset(
    {"car", "truck", "bus", "motorcycle", "bicycle"}
)
SIGN_COCO_LABELS = frozenset({"traffic light", "stop sign"})
PERSON_COCO_LABELS = frozenset({"person"})

# Pack registry: name → metadata.
# source: fiftyone_coco | fiftyone_coco_small | synthetic_* 
PACKS: dict[str, dict[str, Any]] = {
    # --- Core ---
    "objects": {
        "layer": "core",
        "source": "fiftyone_coco",
        "target_name": "objects",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    "vehicles": {
        "layer": "core",
        "source": "fiftyone_coco",
        "target_name": "vehicles",
        "tier": "B",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": sorted(VEHICLE_COCO_LABELS),
        "label_map": None,
    },
    "ocr_plates": {
        "layer": "core",
        "source": "synthetic_plates",
        "target_name": "ocr",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    # --- Extended ---
    "signs": {
        "layer": "extended",
        "source": "fiftyone_coco",
        "target_name": "signs",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": sorted(SIGN_COCO_LABELS),
        "label_map": None,
    },
    "faces": {
        "layer": "extended",
        "source": "synthetic_faces",
        "target_name": "faces",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    "pose": {
        "layer": "extended",
        "source": "fiftyone_coco",
        "target_name": "pose",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": sorted(PERSON_COCO_LABELS),
        "label_map": None,
    },
    "pedestrians": {
        "layer": "extended",
        "source": "fiftyone_coco",
        "target_name": "pedestrians",
        "tier": "B",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": sorted(PERSON_COCO_LABELS),
        "label_map": None,
    },
    "scene": {
        "layer": "extended",
        # No pixel GT masks — honest Tier B schema/smoke on live seg response.
        "source": "fiftyone_coco",
        "target_name": "scene",
        "tier": "B",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
        "require_boxes": False,
    },
    "ocr_text": {
        "layer": "extended",
        "source": "synthetic_text",
        "target_name": "ocr",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    # --- Experimental ---
    "instances": {
        "layer": "experimental",
        "source": "fiftyone_coco",
        "target_name": "instances",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    "small_objects": {
        "layer": "experimental",
        "source": "fiftyone_coco_small",
        "target_name": "small_objects",
        "tier": "A",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
        "max_rel_area": 0.05,
    },
    "open_vocab": {
        "layer": "experimental",
        # Prompted OV — honest Tier B (schema) when label taxonomy is open.
        "source": "fiftyone_coco",
        "target_name": "open_vocab",
        "tier": "B",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    "scene_cls": {
        "layer": "experimental",
        "source": "synthetic_scene_cls",
        "target_name": "scene_cls",
        "tier": "B",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    "anomaly": {
        "layer": "experimental",
        "source": "synthetic_anomaly",
        "target_name": "anomaly",
        "tier": "C",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
    "face_id": {
        "layer": "experimental",
        "source": "synthetic_face_id",
        "target_name": "face_id",
        "tier": "B",
        "max_samples": DEFAULT_MAX_SAMPLES,
        "coco_filter": None,
        "label_map": None,
    },
}

CORE_PACKS = tuple(n for n, p in PACKS.items() if p["layer"] == "core")
EXTENDED_PACKS = tuple(n for n, p in PACKS.items() if p["layer"] == "extended")
EXPERIMENTAL_PACKS = tuple(
    n for n, p in PACKS.items() if p["layer"] == "experimental"
)
ALL_PACKS = CORE_PACKS + EXTENDED_PACKS + EXPERIMENTAL_PACKS


def resolve_pack_names(packs_arg: str) -> list[str]:
    """Return suite names for --packs core|all (default core)."""
    key = (packs_arg or "core").strip().lower()
    if key == "core":
        return list(CORE_PACKS)
    if key == "all":
        return list(ALL_PACKS)
    raise ValueError(f"Unknown --packs value: {packs_arg!r} (use core|all)")


def flat_name(suite: str, index: int, ext: str = ".jpg") -> str:
    """Stable root filename: fo_<suite>_NNNN.jpg"""
    prefix = FLAT_PREFIX.get(suite, f"fo_{suite}_")
    return f"{prefix}{index:04d}{ext}"
