"""Detección de rostros vía PaddleX face_detection."""

from detection.faces.client import (
    ENABLE_FACE_DETECTION,
    PADDLEX_FACES_URL,
    infer_faces,
    normalize_face_result,
    reset_face_tracker,
)

__all__ = [
    "ENABLE_FACE_DETECTION",
    "PADDLEX_FACES_URL",
    "infer_faces",
    "normalize_face_result",
    "reset_face_tracker",
]
