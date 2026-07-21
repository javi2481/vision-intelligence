"""Reconocimiento facial (identidad); bbox de rostros sigue en faces/."""

from detection.face_id.client import (
    ENABLE_FACE_ID,
    PADDLEX_FACE_ID_URL,
    infer_face_id,
    normalize_face_id_result,
    reset_face_id_tracker,
)

__all__ = [
    "ENABLE_FACE_ID",
    "PADDLEX_FACE_ID_URL",
    "infer_face_id",
    "normalize_face_id_result",
    "reset_face_id_tracker",
]
