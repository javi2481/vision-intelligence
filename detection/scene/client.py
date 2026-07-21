"""Cliente HTTP al pipeline semantic_segmentation (PaddleX :8085).

Opcional (ENABLE_SCENE_SEG). Deriva scene_type (street/highway/…) e
infraestructura vial desde labelMap. Modos:
  cityscapes — escena urbana (default)
  lane — 4 clases PP-Vehicle (bg / double_yellow / solid / dashed)
  bdd_marks — categorías BDD lane marking (incl. crosswalk)

Caída aislada: no degrada el bridge.
"""

from __future__ import annotations

import base64
import logging
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

logger = logging.getLogger("detection.scene")

PADDLEX_SCENE_URL = os.getenv("PADDLEX_SCENE_URL", "http://paddlex-scene:8085")
PADDLEX_SCENE_PREDICT_PATH = os.getenv(
    "PADDLEX_SCENE_PREDICT_PATH", "/semantic-segmentation"
)
ENABLE_SCENE_SEG = os.getenv("ENABLE_SCENE_SEG", "false").strip().lower() in (
    "1",
    "true",
    "yes",
)
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
SCENE_LABEL_MODE = os.getenv("SCENE_LABEL_MODE", "cityscapes").strip().lower()
CROSSWALK_MIN_RATIO = float(os.getenv("CROSSWALK_MIN_RATIO", "0.01"))

CITYSCAPES_TRAIN_ID_NAMES: dict[int, str] = {
    0: "road",
    1: "sidewalk",
    2: "building",
    3: "wall",
    4: "fence",
    5: "pole",
    6: "traffic light",
    7: "traffic sign",
    8: "vegetation",
    9: "terrain",
    10: "sky",
    11: "person",
    12: "rider",
    13: "car",
    14: "truck",
    15: "bus",
    16: "train",
    17: "motorcycle",
    18: "bicycle",
}

# PP-Vehicle / PP-LiteSeg BDD 4-class lane
LANE_ID_NAMES: dict[int, str] = {
    0: "background",
    1: "double_yellow",
    2: "solid",
    3: "dashed",
}

# BDD100K lane marking categories (trainId)
BDD_MARKS_ID_NAMES: dict[int, str] = {
    0: "crosswalk",
    1: "double_other",
    2: "double_white",
    3: "double_yellow",
    4: "road_curb",
    5: "single_other",
    6: "single_white",
    7: "single_yellow",
    8: "background",
}


def _extract_label_map(result: dict[str, Any]) -> Optional[list[int]]:
    """Saca labelMap / pred aplanado desde shapes típicos del serving."""
    for key in ("labelMap", "label_map", "pred"):
        raw = result.get(key)
        if raw is None:
            continue
        if isinstance(raw, list):
            flat: list[int] = []
            if raw and isinstance(raw[0], list):
                for row in raw:
                    if isinstance(row, list):
                        flat.extend(int(v) for v in row)
            else:
                flat = [int(v) for v in raw]
            return flat or None
    return None


def class_ratios_from_label_map(
    label_map: list[int],
    id_to_name: dict[int, str],
) -> dict[str, float]:
    """Ratios de píxeles por nombre de clase (0..1). Ignora ids desconocidos."""
    if not label_map:
        return {}
    counts: Counter[int] = Counter(label_map)
    total = float(len(label_map))
    ratios: dict[str, float] = {}
    for cid, count in counts.items():
        name = id_to_name.get(int(cid))
        if not name or name == "background":
            continue
        ratios[name] = round(count / total, 4)
    return ratios


def infer_scene_type(ratios: dict[str, float]) -> tuple[str, float]:
    """Heurística street/highway/parking/rural/unknown a partir de ratios."""
    road = ratios.get("road", 0.0)
    sidewalk = ratios.get("sidewalk", 0.0)
    building = ratios.get("building", 0.0)
    vegetation = ratios.get("vegetation", 0.0)
    terrain = ratios.get("terrain", 0.0)
    sky = ratios.get("sky", 0.0)

    if road < 0.05 and building < 0.05 and sidewalk < 0.02:
        return "unknown", 0.3

    if road >= 0.25 and sidewalk < 0.04 and building < 0.12:
        conf = min(1.0, 0.5 + road + (0.1 - sidewalk) + (0.15 - building) * 0.5)
        return "highway", round(max(0.4, conf), 3)

    if road >= 0.08 and (sidewalk >= 0.02 or building >= 0.1):
        conf = min(1.0, 0.45 + road * 0.5 + sidewalk + building * 0.3)
        return "street", round(max(0.4, conf), 3)

    if road >= 0.15 and building < 0.08 and sky < 0.35:
        return "parking", 0.55

    if (vegetation + terrain) >= 0.35 and building < 0.1:
        return "rural", round(min(1.0, 0.4 + vegetation + terrain), 3)

    if road >= 0.1:
        return "street", 0.45
    return "unknown", 0.35


def build_infra(ratios: dict[str, float]) -> dict[str, Any]:
    """Flags y ratios de infraestructura vial relevantes."""
    return {
        "has_road": ratios.get("road", 0.0) >= 0.05,
        "has_sidewalk": ratios.get("sidewalk", 0.0) >= 0.015,
        "road_ratio": ratios.get("road", 0.0),
        "sidewalk_ratio": ratios.get("sidewalk", 0.0),
        "building_ratio": ratios.get("building", 0.0),
        "traffic_sign_ratio": ratios.get("traffic sign", 0.0),
        "traffic_light_ratio": ratios.get("traffic light", 0.0),
    }


def build_lanes_from_ratios(ratios: dict[str, float]) -> dict[str, Any]:
    """Resumen de marcas viales (modos lane y bdd_marks)."""
    solid = (
        ratios.get("solid", 0.0)
        + ratios.get("single_white", 0.0)
        + ratios.get("single_yellow", 0.0)
        + ratios.get("single_other", 0.0)
    )
    dashed = ratios.get("dashed", 0.0)
    double_yellow = ratios.get("double_yellow", 0.0) + ratios.get(
        "double_white", 0.0
    ) + ratios.get("double_other", 0.0)
    coverage = round(solid + dashed + double_yellow, 4)
    return {
        "solid": round(solid, 4),
        "dashed": round(dashed, 4),
        "double_yellow": round(double_yellow, 4),
        "coverage": coverage,
        "present": coverage >= 0.002,
    }


def build_crosswalk_from_ratios(
    ratios: dict[str, float],
    *,
    min_ratio: float = CROSSWALK_MIN_RATIO,
) -> dict[str, Any]:
    """Post-proceso: presencia de crosswalk por ratio de píxeles."""
    ratio = float(ratios.get("crosswalk", 0.0))
    return {
        "present": ratio >= min_ratio,
        "ratio": round(ratio, 4),
        "min_ratio": min_ratio,
    }


def normalize_scene_result(
    data: dict[str, Any],
    *,
    frame_wh: Optional[tuple[int, int]] = None,
    label_mode: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Traduce semantic_segmentation → una det entity_type=scene."""
    mode = (label_mode or SCENE_LABEL_MODE).strip().lower()
    result = data.get("result", data)
    if not isinstance(result, dict):
        if isinstance(result, list) and result and isinstance(result[0], dict):
            result = result[0]
        else:
            return None

    label_map = _extract_label_map(result)
    if not label_map:
        return None

    lanes: Optional[dict[str, Any]] = None
    crosswalk: Optional[dict[str, Any]] = None

    if mode == "lane":
        ratios = class_ratios_from_label_map(label_map, LANE_ID_NAMES)
        lanes = build_lanes_from_ratios(ratios)
        scene_type = "unknown"
        confidence = 0.5 if lanes.get("present") else 0.3
        infra: dict[str, Any] = {"has_road": False, "has_sidewalk": False}
    elif mode == "bdd_marks":
        id_names = dict(BDD_MARKS_ID_NAMES)
        names = result.get("labelNames") or result.get("label_names")
        if isinstance(names, list) and names:
            id_names = {i: str(n).replace(" ", "_").lower() for i, n in enumerate(names)}
        ratios = class_ratios_from_label_map(label_map, id_names)
        lanes = build_lanes_from_ratios(ratios)
        crosswalk = build_crosswalk_from_ratios(ratios)
        scene_type = "street" if (lanes and lanes.get("present")) or (
            crosswalk and crosswalk.get("present")
        ) else "unknown"
        confidence = 0.55 if scene_type == "street" else 0.3
        infra = {
            "has_road": True if lanes and lanes.get("present") else False,
            "has_sidewalk": False,
            "has_crosswalk": bool(crosswalk and crosswalk.get("present")),
        }
    else:
        id_names = dict(CITYSCAPES_TRAIN_ID_NAMES)
        names = result.get("labelNames") or result.get("label_names")
        if isinstance(names, list) and names:
            id_names = {i: str(n) for i, n in enumerate(names)}
        ratios = class_ratios_from_label_map(label_map, id_names)
        scene_type, confidence = infer_scene_type(ratios)
        infra = build_infra(ratios)
        # Si el modelo expone "crosswalk" en label_names (fine-tune), detectarlo
        if "crosswalk" in ratios:
            crosswalk = build_crosswalk_from_ratios(ratios)
            lanes = build_lanes_from_ratios(ratios) if any(
                k in ratios
                for k in ("solid", "dashed", "double_yellow", "single_white")
            ) else None

    if frame_wh is not None:
        w, h = frame_wh
        bbox = [0.0, 0.0, float(w), float(h)]
    else:
        shape = result.get("shape") or result.get("imgSize")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            h, w = int(shape[0]), int(shape[1])
            bbox = [0.0, 0.0, float(w), float(h)]
        else:
            bbox = [0.0, 0.0, 1.0, 1.0]

    now = datetime.now(timezone.utc).isoformat()
    return {
        "track_id": "scene-0",
        "label": scene_type,
        "score": confidence,
        "bbox": bbox,
        "entity_type": "scene",
        "frame_ts": now,
        "scene": {
            "type": scene_type,
            "ratios": ratios,
            "infra": infra,
            "lanes": lanes,
            "crosswalk": crosswalk,
            "label_mode": mode,
        },
    }


async def infer_scene(
    client: httpx.AsyncClient,
    jpeg: bytes,
    *,
    frame_wh: Optional[tuple[int, int]] = None,
) -> Optional[dict[str, Any]]:
    """POST JPEG a semantic-segmentation. None ante fallo o sin máscara."""
    if not ENABLE_SCENE_SEG:
        return None
    url = f"{PADDLEX_SCENE_URL.rstrip('/')}{PADDLEX_SCENE_PREDICT_PATH}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Scene seg infer error (isolated): %s", exc)
        return None

    if not isinstance(data, dict):
        return None
    if data.get("errorCode") not in (None, 0, "0"):
        logger.debug("Scene seg error: %s", data.get("errorMsg"))
        return None
    return normalize_scene_result(data, frame_wh=frame_wh)
