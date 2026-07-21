"""Overlay local de preview: bboxes + labels EN sobre el frame (sin CJK de PaddleX).

No usa result.image del serving. Colores por tipo de vehículo o fallback por track_id.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2

from detection.common.geometry import encode_jpeg

_PREVIEW_TYPE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "sedan": (255, 160, 0),
    "suv": (0, 140, 255),
    "van": (200, 0, 200),
    "truck": (0, 215, 255),
    "bus": (0, 0, 255),
    "mpv": (255, 0, 128),
    "pickup": (0, 200, 120),
    "hatchback": (180, 100, 0),
    "car": (255, 160, 0),
    "vehicle": (80, 80, 255),
}

_PREVIEW_FALLBACK_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (255, 160, 0),
    (0, 140, 255),
    (200, 0, 200),
    (0, 215, 255),
    (0, 0, 255),
    (255, 0, 128),
    (0, 200, 120),
    (180, 100, 0),
)


def preview_label(det: dict[str, Any]) -> str:
    """Texto de etiqueta en inglés para el overlay (tipo + color / attrs)."""
    entity = str(det.get("entity_type") or "").strip().lower()
    if entity == "scene":
        scene = det.get("scene") if isinstance(det.get("scene"), dict) else {}
        stype = (scene or {}).get("type") or det.get("label") or "scene"
        parts = [f"scene:{stype}"]
        cw = (scene or {}).get("crosswalk") if isinstance(scene, dict) else None
        if isinstance(cw, dict) and cw.get("present"):
            parts.append("xwalk")
        lanes = (scene or {}).get("lanes") if isinstance(scene, dict) else None
        if isinstance(lanes, dict) and lanes.get("present"):
            parts.append("lanes")
        return " ".join(parts)
    if entity == "face":
        return "face"
    if entity == "face_id":
        return f"id:{det.get('identity') or det.get('label') or '?'}"
    if entity == "text":
        t = str(det.get("text") or det.get("label") or "text")
        return t[:24]
    if entity == "pose":
        return "pose"
    if entity == "sign":
        return f"sign:{det.get('label') or '?'}"
    if entity in {"scene_cls", "anomaly", "instance", "small_object", "open_vocab"}:
        return f"{entity}:{det.get('label') or '?'}"

    parts: list[str] = []
    label = det.get("label")
    color = det.get("color")
    if label:
        parts.append(str(label))
    if color:
        parts.append(str(color))
    person = det.get("person") if isinstance(det.get("person"), dict) else None
    if person:
        gender = person.get("gender")
        age = person.get("age_group")
        if gender:
            parts.append(str(gender)[0].upper())
        if age:
            parts.append(str(age))
    return " ".join(parts) or "vehicle"


def preview_box_color(det: dict[str, Any]) -> tuple[int, int, int]:
    """Color BGR estable por tipo de entidad / vehículo."""
    entity = str(det.get("entity_type") or "").strip().lower()
    entity_colors = {
        "face": (0, 200, 255),
        "face_id": (0, 165, 255),
        "scene": (180, 180, 180),
        "pose": (255, 100, 50),
        "text": (200, 200, 50),
        "sign": (50, 50, 220),
        "scene_cls": (160, 160, 100),
        "instance": (100, 180, 100),
        "small_object": (100, 100, 255),
        "anomaly": (0, 0, 220),
        "open_vocab": (180, 100, 180),
    }
    if entity in entity_colors:
        return entity_colors[entity]
    label = str(det.get("label") or "").strip().lower()
    if label == "person":
        return (80, 200, 80)
    if label in _PREVIEW_TYPE_COLORS_BGR:
        return _PREVIEW_TYPE_COLORS_BGR[label]
    tid = str(det.get("track_id") or "0")
    digits = "".join(ch for ch in tid if ch.isdigit()) or "0"
    idx = int(digits) % len(_PREVIEW_FALLBACK_PALETTE_BGR)
    return _PREVIEW_FALLBACK_PALETTE_BGR[idx]


def draw_preview(frame, detections: list[dict[str, Any]]) -> Optional[bytes]:
    """Dibuja bboxes + labels EN y badge de escena. None si encode falla."""
    canvas = frame.copy()
    for det in detections or []:
        entity = str(det.get("entity_type") or "").strip().lower()
        if entity == "scene":
            scene = det.get("scene") if isinstance(det.get("scene"), dict) else {}
            stype = (scene or {}).get("type") or det.get("label") or "scene"
            badge = f"scene:{stype}"
            cv2.rectangle(canvas, (8, 8), (8 + 12 * len(badge), 36), (40, 40, 40), -1)
            cv2.putText(
                canvas,
                badge,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            continue

        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1 = int(round(float(bbox[0])))
        y1 = int(round(float(bbox[1])))
        x2 = int(round(float(bbox[2])))
        y2 = int(round(float(bbox[3])))
        color = preview_box_color(det)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        text = preview_label(det)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 4
        label_y1 = max(0, y1 - th - baseline - pad * 2)
        label_y2 = max(th + baseline + pad * 2, y1)
        label_x2 = min(canvas.shape[1] - 1, x1 + tw + pad * 2)
        cv2.rectangle(
            canvas, (x1, label_y1), (label_x2, label_y2), color, thickness=-1
        )
        cv2.putText(
            canvas,
            text,
            (x1 + pad, label_y2 - baseline - pad),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return encode_jpeg(canvas)
