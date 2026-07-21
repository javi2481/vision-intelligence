"""Registro de capacidades del bridge (infer + merge + reset).

Agregar una capacidad nueva: registrar aquí + carpeta detection/<cap>/.
No hace falta tocar el unpack manual de asyncio.gather en bridge/main.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, Optional

import httpx

from detection.anomaly import ENABLE_ANOMALY, infer_anomaly
from detection.face_id import ENABLE_FACE_ID, infer_face_id, reset_face_id_tracker
from detection.faces import (
    ENABLE_FACE_DETECTION,
    PADDLEX_FACES_URL,
    infer_faces,
    reset_face_tracker,
)
from detection.instances import ENABLE_INSTANCE_SEG, infer_instances
from detection.objects import (
    attach_object_track_ids,
    infer_objects,
    merge_coco_detections,
    reset_object_tracker,
)
from detection.open_vocab import ENABLE_OPEN_VOCAB, infer_open_vocab
from detection.pedestrians import (
    ENABLE_PEDESTRIAN_ATTRS,
    PADDLEX_PEDESTRIANS_URL,
    infer_pedestrian_attrs,
    merge_person_attributes,
)
from detection.plates import ENABLE_PLATE_OCR, PADDLEX_OCR_URL
from detection.pose import ENABLE_POSE, infer_pose, reset_pose_tracker
from detection.scene import ENABLE_SCENE_SEG, PADDLEX_SCENE_URL, infer_scene
from detection.scene_cls import ENABLE_SCENE_CLS, infer_scene_cls
from detection.signs import ENABLE_SIGNS, infer_signs, reset_signs_tracker
from detection.small_objects import ENABLE_SMALL_OBJECTS, infer_small_objects
from detection.text import ENABLE_SCENE_OCR, infer_scene_ocr
from detection.vehicles import infer_vehicles, reset_vehicle_tracker
from detection.vehicles import client as vehicles_client

MergeKind = Literal[
    "vehicles",
    "objects",
    "ped_attrs",
    "extend_scaled",
    "append_one",
]

InferFn = Callable[..., Awaitable[Any]]
ResetFn = Callable[[], None]


@dataclass(frozen=True)
class Capability:
    name: str
    infer: InferFn
    merge: MergeKind
    reset: Optional[ResetFn] = None
    critical: bool = False
    """Si True y el resultado es None → bridge degraded (solo vehicles)."""
    needs_frame_wh: bool = False


def _infer_scene(client: httpx.AsyncClient, jpeg: bytes, *, frame_wh: tuple[int, int]):
    return infer_scene(client, jpeg, frame_wh=frame_wh)


CAPABILITIES: list[Capability] = [
    Capability(
        "vehicles",
        infer_vehicles,
        "vehicles",
        reset=reset_vehicle_tracker,
        critical=True,
    ),
    Capability("objects", infer_objects, "objects", reset=reset_object_tracker),
    Capability("faces", infer_faces, "extend_scaled", reset=reset_face_tracker),
    Capability("pedestrians", infer_pedestrian_attrs, "ped_attrs"),
    Capability(
        "scene",
        _infer_scene,
        "append_one",
        needs_frame_wh=True,
    ),
    Capability("pose", infer_pose, "extend_scaled", reset=reset_pose_tracker),
    Capability("text", infer_scene_ocr, "extend_scaled"),
    Capability("face_id", infer_face_id, "extend_scaled", reset=reset_face_id_tracker),
    Capability("signs", infer_signs, "extend_scaled", reset=reset_signs_tracker),
    Capability("scene_cls", infer_scene_cls, "append_one"),
    Capability("instances", infer_instances, "extend_scaled"),
    Capability("small_objects", infer_small_objects, "extend_scaled"),
    Capability("anomaly", infer_anomaly, "append_one"),
    Capability("open_vocab", infer_open_vocab, "extend_scaled"),
]


def reset_all_trackers() -> None:
    for cap in CAPABILITIES:
        if cap.reset is not None:
            cap.reset()


def capability_status_line() -> str:
    """Resumen compacto para el log de arranque del bridge."""
    return (
        f"paddlex={vehicles_client.PADDLEX_URL}{vehicles_client.PADDLEX_PREDICT_PATH} "
        f"ocr={PADDLEX_OCR_URL if ENABLE_PLATE_OCR or ENABLE_SCENE_OCR else 'off'} "
        f"faces={PADDLEX_FACES_URL if ENABLE_FACE_DETECTION else 'off'} "
        f"ped={PADDLEX_PEDESTRIANS_URL if ENABLE_PEDESTRIAN_ATTRS else 'off'} "
        f"scene={PADDLEX_SCENE_URL if ENABLE_SCENE_SEG else 'off'} "
        f"pose={ENABLE_POSE} text={ENABLE_SCENE_OCR} face_id={ENABLE_FACE_ID} "
        f"signs={ENABLE_SIGNS} exp[scene_cls={ENABLE_SCENE_CLS} "
        f"inst={ENABLE_INSTANCE_SEG} small={ENABLE_SMALL_OBJECTS} "
        f"anom={ENABLE_ANOMALY} ov={ENABLE_OPEN_VOCAB}]"
    )


# Re-export merge helpers used by bridge after gather
__all__ = [
    "CAPABILITIES",
    "Capability",
    "attach_object_track_ids",
    "capability_status_line",
    "merge_coco_detections",
    "merge_person_attributes",
    "reset_all_trackers",
]
