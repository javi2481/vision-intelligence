"""Detección de vehículos + atributos (tipo/color) vía PaddleX."""

from detection.vehicles.client import (
    decode_paddlex_result_image,
    infer_vehicles,
    infer_vehicles_tiled_sync,
    normalize_vehicle_result,
    parse_attr_labels,
    parse_vehicle_boxes,
    reset_vehicle_tracker,
)

__all__ = [
    "decode_paddlex_result_image",
    "infer_vehicles",
    "infer_vehicles_tiled_sync",
    "normalize_vehicle_result",
    "parse_attr_labels",
    "parse_vehicle_boxes",
    "reset_vehicle_tracker",
]
