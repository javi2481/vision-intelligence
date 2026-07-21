"""Resolución de fuente foto: poll adapter /media/current → ruta local o idle."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("bridge.media")

MEDIA_DIR = os.getenv("MEDIA_DIR", "/media")
MEDIA_IMAGE_SUBDIR = "images"
MEDIA_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def media_type_by_extension(filename: str) -> Optional[str]:
    """Clasifica filename como 'image' por extensión; None si no matchea."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in MEDIA_IMAGE_EXTENSIONS:
        return "image"
    return None


def is_safe_media_name(name: str) -> bool:
    """Allow-list: basename plano sin path traversal."""
    return bool(name) and os.path.basename(name) == name and name not in (".", "..")


def resolve_media_path(name: str) -> Optional[str]:
    """Ruta absoluta bajo MEDIA_DIR/images para una muestra validada."""
    if not is_safe_media_name(name):
        return None
    return os.path.join(MEDIA_DIR, MEDIA_IMAGE_SUBDIR, name)


def resolve_active_source(selected: Optional[dict[str, Any]]) -> Optional[str]:
    """Ruta absoluta de la foto activa, o None si no hay selección (idle)."""
    if not selected or not selected.get("name"):
        return None
    media_type = selected.get("type") or "image"
    if media_type == "image" or media_type_by_extension(selected["name"]) == "image":
        path = resolve_media_path(selected["name"])
        if path is not None:
            return path
    logger.warning(
        "Selected media ignored (solo imagenes): name=%s type=%s",
        selected.get("name"),
        selected.get("type"),
    )
    return None
