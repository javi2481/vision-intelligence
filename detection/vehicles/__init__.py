"""Detección de vehículos + atributos (tipo/color) vía PaddleX."""

from detection.vehicles.client import (
    decode_paddlex_result_image,
    infer_vehicles,
    normalize_vehicle_result,
    parse_attr_labels,
    reset_vehicle_tracker,
)

__all__ = [
    "decode_paddlex_result_image",
    "infer_vehicles",
    "normalize_vehicle_result",
    "parse_attr_labels",
    "reset_vehicle_tracker",
]
