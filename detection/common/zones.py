"""Zonas poligonales normalizadas [0,1] → PolygonZone en píxeles absolutos.

Denorm: ``pts * [w, h]`` únicamente. **Prohibido** ``sv.denormalize_boxes``
(esa API espera xyxy, no vértices (N,2)).

La máscara de ``PolygonZone`` (supervision ≥0.24) se dimensiona por el
extent del polígono (``x_max+2, y_max+2``), no por el frame: un anchor fuera
del extent es miss aunque el clipping lo acerque al borde.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import supervision as sv

logger = logging.getLogger("detection.zones")

# JSON: [{"id":"no_parking","polygon":[[0.1,0.1],[0.4,0.1],[0.4,0.5],[0.1,0.5]]}]
VI_ZONES_JSON = os.getenv("VI_ZONES_JSON", "").strip()


@dataclass(frozen=True)
class ZoneConfig:
    """Zona con id estable y polígono normalizado [0,1] (N,2)."""

    id: str
    polygon_norm: np.ndarray  # float64 (N,2)


def denormalize_polygon(
    polygon_norm: np.ndarray, frame_wh: tuple[int, int]
) -> np.ndarray:
    """``pts_norm * [w, h]`` → int64 (N,2). No usar denormalize_boxes."""
    w, h = int(frame_wh[0]), int(frame_wh[1])
    pts = np.asarray(polygon_norm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"polygon_norm must be (N,2), got {pts.shape}")
    scale = np.array([w, h], dtype=np.float64)
    return np.rint(pts * scale).astype(np.int64)


def parse_zones_json(raw: str) -> list[ZoneConfig]:
    """Parsea VI_ZONES_JSON; [] si vacío o inválido."""
    if not raw or not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("VI_ZONES_JSON invalid JSON: %s", exc)
        return []
    if not isinstance(payload, list):
        logger.warning("VI_ZONES_JSON must be a list of zone objects")
        return []

    zones: list[ZoneConfig] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        zid = str(item.get("id") or "").strip()
        poly = item.get("polygon")
        if not zid or not isinstance(poly, list) or len(poly) < 3:
            continue
        try:
            arr = np.asarray(poly, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if arr.ndim != 2 or arr.shape[1] != 2:
            continue
        zones.append(ZoneConfig(id=zid, polygon_norm=arr))
    return zones


def load_zone_configs(raw: Optional[str] = None) -> list[ZoneConfig]:
    """Carga zonas desde ``raw`` o env ``VI_ZONES_JSON``."""
    return parse_zones_json(VI_ZONES_JSON if raw is None else raw)


def absolute_polygons(
    zones: list[ZoneConfig], frame_wh: tuple[int, int]
) -> list[tuple[str, np.ndarray]]:
    """Lista (zone_id, polygon_abs int64) para preview / PolygonZone."""
    return [(z.id, denormalize_polygon(z.polygon_norm, frame_wh)) for z in zones]


def tag_detections_with_zones(
    detections: list[dict[str, Any]],
    frame_wh: tuple[int, int],
    zones: list[ZoneConfig],
) -> list[dict[str, Any]]:
    """Añade ``zones: list[str]`` (ids hit) a cada det con bbox. Sin ByteTrack."""
    if not zones or not detections:
        return detections

    # Prebuild zones once per frame.
    built: list[tuple[str, sv.PolygonZone]] = []
    for z in zones:
        poly_abs = denormalize_polygon(z.polygon_norm, frame_wh)
        built.append((z.id, sv.PolygonZone(polygon=poly_abs)))

    out: list[dict[str, Any]] = []
    for det in detections:
        row = dict(det)
        bbox = det.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            out.append(row)
            continue
        xyxy = np.array([bbox], dtype=np.float32)
        conf = np.array([float(det.get("score") or 0.0)], dtype=np.float32)
        # class_id dummy — trigger solo usa anchors de xyxy.
        sv_det = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=np.array([0], dtype=np.int32),
        )
        hits: list[str] = []
        for zid, zone in built:
            mask = zone.trigger(sv_det)
            if bool(mask[0]):
                hits.append(zid)
        if hits:
            row["zones"] = hits
        else:
            row.pop("zones", None)
        out.append(row)
    return out
