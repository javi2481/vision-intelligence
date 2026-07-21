"""Cliente HTTP al pipeline pedestrian_attribute_recognition (PaddleX :8084).

Opcional (ENABLE_PEDESTRIAN_ATTRS). Enriquece dets COCO `person` por IoU;
no es un detector de personas aparte (el bbox vive en objects/).
Caída aislada: no degrada el bridge.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from detection.common.tracking import iou

logger = logging.getLogger("detection.pedestrians")

PADDLEX_PEDESTRIANS_URL = os.getenv(
    "PADDLEX_PEDESTRIANS_URL", "http://paddlex-pedestrians:8084"
)
PADDLEX_PEDESTRIANS_PREDICT_PATH = os.getenv(
    "PADDLEX_PEDESTRIANS_PREDICT_PATH", "/pedestrian-attribute-recognition"
)
ENABLE_PEDESTRIAN_ATTRS = os.getenv(
    "ENABLE_PEDESTRIAN_ATTRS", "false"
).strip().lower() in ("1", "true", "yes")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "30.0"))
PERSON_ATTR_IOU_THRESHOLD = float(os.getenv("PERSON_ATTR_IOU_THRESHOLD", "0.3"))


def _latin_token(raw: Any) -> str:
    """Parte latina antes de paréntesis CJK; minúsculas."""
    text = str(raw).split("(")[0].strip().lower()
    return re.sub(r"[\u4e00-\u9fff]+", "", text).strip()


def parse_person_attributes(
    labels: list[Any], scores: list[Any]
) -> dict[str, Any]:
    """Normaliza labels de attrs PP-Human a un dict EN compacto.

    Heurística por keywords (gender/age/color/direction); el resto va a `other`.
    """
    attrs: dict[str, Any] = {}
    other: list[str] = []
    gender_keys = {"male", "female", "man", "woman"}
    age_keys = {
        "child", "adult", "elderly", "teenager", "young", "old",
        "age16-30", "age31-45", "age46-60", "ageabove60", "agebelow16",
    }
    direction_keys = {
        "front", "back", "side", "left", "right", "frontside", "backside",
    }

    for label, score in zip(labels or [], scores or []):
        token = _latin_token(label)
        if not token:
            continue
        score_f = float(score or 0.0)
        if token in gender_keys and "gender" not in attrs:
            attrs["gender"] = "female" if token in {"female", "woman"} else "male"
            attrs["gender_score"] = score_f
        elif any(token.startswith(a) or token == a for a in age_keys) and (
            "age_group" not in attrs
        ):
            attrs["age_group"] = token
            attrs["age_group_score"] = score_f
        elif token in direction_keys and "direction" not in attrs:
            attrs["direction"] = token
            attrs["direction_score"] = score_f
        elif "color" in token or token in {
            "red", "blue", "green", "yellow", "white", "black",
            "brown", "grey", "gray", "orange", "purple",
        }:
            if "upper_color" not in attrs:
                attrs["upper_color"] = token.replace("upper_", "").replace(
                    "lower_", ""
                )
        else:
            other.append(token)

    if other:
        attrs["other"] = other[:12]
    return attrs


def normalize_pedestrian_result(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Traduce pedestrian_attribute_recognition → dets crudas con bbox+person."""
    result = data.get("result", data)
    boxes: list[dict[str, Any]] = []
    if isinstance(result, dict):
        raw = (
            result.get("boxes")
            or result.get("pedestrians")
            or result.get("persons")
            or []
        )
        if isinstance(raw, list):
            boxes = raw
    elif isinstance(result, list):
        for item in result:
            if isinstance(item, dict) and (
                "boxes" in item or "pedestrians" in item
            ):
                boxes.extend(item.get("boxes") or item.get("pedestrians") or [])
            elif isinstance(item, dict) and (
                "coordinate" in item or "bbox" in item
            ):
                boxes.append(item)

    detections: list[dict[str, Any]] = []
    for box in boxes:
        if not isinstance(box, dict):
            continue
        coord = box.get("coordinate") or box.get("bbox")
        if not coord or len(coord) < 4:
            continue
        bbox = [float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])]

        labels: list[Any] = []
        scores: list[Any] = []
        attrs_raw = box.get("attributes")
        if isinstance(attrs_raw, list):
            for a in attrs_raw:
                if isinstance(a, dict):
                    labels.append(a.get("label") or a.get("name") or "")
                    scores.append(a.get("score") or a.get("confidence") or 0.0)
                else:
                    labels.append(a)
                    scores.append(1.0)
        else:
            labels = list(box.get("labels") or [])
            scores = list(box.get("cls_scores") or box.get("scores") or [])

        person = parse_person_attributes(labels, scores)
        detections.append(
            {
                "label": "person",
                "score": float(
                    box.get("score") or box.get("det_score") or 0.0
                ),
                "bbox": bbox,
                "person": person,
                "entity_type": "object",
            }
        )
    return detections


def merge_person_attributes(
    object_dets: list[dict[str, Any]],
    attr_dets: list[dict[str, Any]],
    iou_threshold: float = PERSON_ATTR_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Enriquece dets COCO person con attrs; añade attrs sin match como person.

    Mutates matching object dets in place (añade clave `person`). Devuelve la
    lista object_dets (misma referencia) más attrs huérfanos con track sintético.
    """
    if not attr_dets:
        return list(object_dets)

    persons = [
        d
        for d in object_dets
        if str(d.get("label") or "").strip().lower() == "person" and d.get("bbox")
    ]
    used_attr: set[int] = set()
    for person_det in persons:
        best_i = -1
        best_iou = iou_threshold
        pb = person_det["bbox"]
        for i, attr in enumerate(attr_dets):
            if i in used_attr:
                continue
            ab = attr.get("bbox")
            if not ab:
                continue
            score = iou(pb, ab)
            if score > best_iou:
                best_iou = score
                best_i = i
        if best_i >= 0:
            used_attr.add(best_i)
            person_det["person"] = dict(attr_dets[best_i].get("person") or {})

    extras: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for i, attr in enumerate(attr_dets):
        if i in used_attr:
            continue
        extras.append(
            {
                "track_id": f"p-attr-{i}",
                "label": "person",
                "score": float(attr.get("score") or 0.0),
                "bbox": attr.get("bbox"),
                "entity_type": "object",
                "person": dict(attr.get("person") or {}),
                "frame_ts": now,
            }
        )
    return list(object_dets) + extras


async def infer_pedestrian_attrs(
    client: httpx.AsyncClient, jpeg: bytes
) -> Optional[list[dict[str, Any]]]:
    """POST JPEG a pedestrian-attribute-recognition. None ante fallo."""
    if not ENABLE_PEDESTRIAN_ATTRS:
        return None
    url = (
        f"{PADDLEX_PEDESTRIANS_URL.rstrip('/')}"
        f"{PADDLEX_PEDESTRIANS_PREDICT_PATH}"
    )
    b64 = base64.b64encode(jpeg).decode("ascii")
    try:
        resp = await client.post(
            url, json={"image": b64}, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Pedestrian attrs infer error (isolated): %s", exc)
        return None

    if not isinstance(data, dict):
        return []
    if data.get("errorCode") not in (None, 0, "0"):
        logger.debug("Pedestrian attrs error: %s", data.get("errorMsg"))
        return None
    return normalize_pedestrian_result(data)
