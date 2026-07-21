"""Atributos de personas (enriquece COCO person; no es detector aparte)."""

from detection.pedestrians.client import (
    ENABLE_PEDESTRIAN_ATTRS,
    PADDLEX_PEDESTRIANS_URL,
    infer_pedestrian_attrs,
    merge_person_attributes,
    normalize_pedestrian_result,
    parse_person_attributes,
)

__all__ = [
    "ENABLE_PEDESTRIAN_ATTRS",
    "PADDLEX_PEDESTRIANS_URL",
    "infer_pedestrian_attrs",
    "merge_person_attributes",
    "normalize_pedestrian_result",
    "parse_person_attributes",
]
