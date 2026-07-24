"""
epp-core contract (schema_version=\"1.0\").

Ubicación: adapter/epp_core.py — contrato portable del Producto B
(Vision Intelligence, EPP v4.6 Punto 12).

Única pieza portable: entra un dict (detección del bridge), sale un PerceptionEvent.
Sin lógica de reglas de negocio; solo normalización y consolidación de tracks.

Payload tipado por entity_type (discriminated union Pydantic v2): cada variante
declara solo sus campos. Identidades (face_id) se pseudonimizan con HMAC-SHA256
antes de salir del edge (Ley 25.326 / edge-first metadata-only).

Asumción del JSON de detección (por ítem)::

    {
      \"track_id\": \"v-42\",
      \"label\": \"car\",
      \"score\": 0.91,
      \"bbox\": [x1, y1, x2, y2],
      \"color\": \"white\",
      \"plate\": {\"text\": \"ABC123\", \"score\": 0.87},
      \"entity_type\": \"vehicle\",
      \"frame_ts\": \"2026-07-18T15:00:00.123Z\"
    }
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("epp_core")

SCHEMA_VERSION = "1.0"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def pseudonymize_identity(raw: Optional[str]) -> Optional[str]:
    """HMAC-SHA256(identity, IDENTITY_HASH_SALT). Nunca emite identity en claro.

    Si hay identity pero falta el salt, se omite el campo (no se filtra plaintext).
    """
    if raw is None or str(raw).strip() == "":
        return None
    salt = os.getenv("IDENTITY_HASH_SALT", "").strip()
    if not salt:
        logger.warning(
            "identity omitted: set IDENTITY_HASH_SALT when ENABLE_FACE_ID is on"
        )
        return None
    digest = hmac.new(
        salt.encode("utf-8"),
        str(raw).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return digest


class Location(BaseModel):
    """Ubicación GeoJSON RFC 7946 (Point). Coordenadas [lon, lat]."""

    type: str = "Point"
    coordinates: list[float] = Field(
        ...,
        min_length=2,
        max_length=3,
        description="[longitude, latitude] o [lon, lat, alt]",
    )


class VehiclePayload(BaseModel):
    entity_type: Literal["vehicle"] = "vehicle"
    color: Optional[str] = None
    vehicle_type: Optional[str] = None
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None
    speed_kmh: Optional[float] = None


class ObjectPayload(BaseModel):
    entity_type: Literal["object"] = "object"
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    speed_kmh: Optional[float] = None
    person: Optional[dict[str, Any]] = None


class FacePayload(BaseModel):
    entity_type: Literal["face"] = "face"
    class_name: str = "face"
    bbox: Optional[list[float]] = None


class ScenePayload(BaseModel):
    entity_type: Literal["scene"] = "scene"
    class_name: Optional[str] = None
    scene_type: Optional[str] = None
    scene: Optional[dict[str, Any]] = None
    bbox: Optional[list[float]] = None


class PosePayload(BaseModel):
    entity_type: Literal["pose"] = "pose"
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    keypoints: Optional[list[Any]] = None


class TextPayload(BaseModel):
    entity_type: Literal["text"] = "text"
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    text: Optional[str] = None


class IdentityPayload(BaseModel):
    """identity ya viene pseudonimizada (HMAC hex)."""

    entity_type: Literal["face_id"] = "face_id"
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    identity: Optional[str] = None


class GenericPayload(BaseModel):
    """sign, scene_cls, instance, small_object, anomaly, open_vocab.

    `entity_type` no tiene default: cada consolidador debe pasarlo
    explícitamente (es el discriminador entre las 6 variantes genéricas).
    """

    entity_type: Literal[
        "sign", "scene_cls", "instance", "small_object", "anomaly", "open_vocab"
    ]
    class_name: Optional[str] = None
    bbox: Optional[list[float]] = None
    text: Optional[str] = None
    identity: Optional[str] = None
    keypoints: Optional[list[Any]] = None


# Union discriminada por `entity_type`: pydantic elige la variante exacta por
# el valor del campo (dict/JSON), en vez de "smart mode" (probar cada tipo por
# forma de campos), que podía instanciar la clase equivocada cuando dos
# variantes comparten forma (p.ej. GenericPayload vs TextPayload).
EntityPayload = Annotated[
    Union[
        VehiclePayload,
        ObjectPayload,
        FacePayload,
        ScenePayload,
        PosePayload,
        TextPayload,
        IdentityPayload,
        GenericPayload,
    ],
    Field(discriminator="entity_type"),
]


class PerceptionEvent(BaseModel):
    """
    Sobre común epp-core 1.0.

    Garantías:
      #1 Dos tiempos: occurred_at (frame) y observed_at (procesamiento).
      #3 Confianza comparable en [0.0, 1.0].
      #4 Pistas, no veredictos: candidate_ids.
      #6 Versionado: schema_version=\"1.0\".
      Payload discriminado por entity_type (Annotated Union + discriminator;
      ver `EntityPayload`). El propio `payload.entity_type` es la fuente de
      verdad de la variante — no hay validación cruzada manual contra el
      `entity_type` del sobre: pydantic ya resuelve/valida la clase concreta
      al deserializar via el discriminador.
    """

    schema_version: str = SCHEMA_VERSION
    entity_type: str = "vehicle"
    occurred_at: datetime
    observed_at: datetime = Field(default_factory=_utc_now)
    confidence: float = Field(..., ge=0.0, le=1.0)
    candidate_ids: list[str] = Field(default_factory=list)
    location: Optional[Location] = None
    # Aditivo SCHEMA 1.0: clientes viejos ignoran ``zones``; no bump de versión.
    zones: Optional[list[str]] = None
    payload: EntityPayload

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @classmethod
    def consolidate_and_emit(
        cls,
        paddlex_detections: list[dict[str, Any]],
        *,
        location: Optional[Location] = None,
    ) -> list[PerceptionEvent]:
        """Agrupa por track_id y emite un PerceptionEvent consolidado por track."""
        if not paddlex_detections:
            return []

        by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for raw in paddlex_detections:
            normalized = _normalize_detection(raw)
            track_id = normalized.get("track_id")
            if not track_id:
                continue
            by_track[str(track_id)].append(normalized)

        events: list[PerceptionEvent] = []
        observed = _utc_now()

        for track_id, detections in by_track.items():
            event = _emit_track(cls, track_id, detections, observed, location)
            if event is not None:
                events.append(event)

        return events


def _normalize_detection(raw: dict[str, Any]) -> dict[str, Any]:
    """Normaliza variantes de campos del JSON PaddleX a un dict interno."""
    plate = raw.get("plate") if isinstance(raw.get("plate"), dict) else {}
    score = raw.get("score", raw.get("conf", raw.get("confidence", 0.0)))
    plate_text = (
        raw.get("plate_text")
        or plate.get("text")
        or plate.get("rec_text")
        or raw.get("plate_number")
    )
    plate_score = (
        raw.get("plate_score")
        or plate.get("score")
        or plate.get("rec_score")
        or score
    )
    bbox = raw.get("bbox") or raw.get("boxes") or raw.get("coordinate")
    frame_ts = raw.get("frame_ts") or raw.get("timestamp") or raw.get("occurred_at")

    return {
        "track_id": raw.get("track_id") or raw.get("tracker_id") or raw.get("id"),
        "label": raw.get("label") or raw.get("vehicle_type") or raw.get("cls"),
        "score": float(score or 0.0),
        "color": raw.get("color") or raw.get("vehicle_color"),
        "plate_text": plate_text,
        "plate_score": float(plate_score or 0.0) if plate_text else None,
        "bbox": list(bbox) if bbox is not None else None,
        "speed_kmh": raw.get("speed_kmh") or raw.get("speed"),
        "frame_ts": frame_ts,
        "entity_type": raw.get("entity_type") or "vehicle",
        "person": raw.get("person") if isinstance(raw.get("person"), dict) else None,
        "scene": raw.get("scene") if isinstance(raw.get("scene"), dict) else None,
        "text": raw.get("text"),
        "identity": raw.get("identity"),
        "keypoints": raw.get("keypoints") if isinstance(raw.get("keypoints"), list) else None,
        "zones": (
            [str(z) for z in raw["zones"]]
            if isinstance(raw.get("zones"), list)
            else None
        ),
    }


def _weighted_vote(
    detections: list[dict[str, Any]],
    value_key: str,
    weight_key: str,
) -> tuple[Optional[str], float]:
    weights: dict[str, float] = defaultdict(float)
    total = 0.0

    for det in detections:
        value = det.get(value_key)
        if value is None or value == "":
            continue
        key = str(value).strip().upper() if value_key == "plate_text" else str(value)
        w = float(det.get(weight_key) or 0.0)
        if w <= 0.0:
            continue
        weights[key] += w
        total += w

    if not weights or total <= 0.0:
        return None, 0.0

    winner, winner_w = max(weights.items(), key=lambda item: item[1])
    return winner, winner_w / total


def _parse_occurred_at(detections: list[dict[str, Any]]) -> datetime:
    latest: Optional[datetime] = None
    for det in detections:
        ts = det.get("frame_ts")
        if not ts:
            continue
        if isinstance(ts, datetime):
            parsed = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        else:
            try:
                parsed = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            except ValueError:
                continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest or _utc_now()


def _collect_zones(detections: list[dict[str, Any]]) -> Optional[list[str]]:
    """Une ids de zona hit en el track (orden de primera aparición)."""
    seen: list[str] = []
    for det in detections:
        zones = det.get("zones")
        if not isinstance(zones, list):
            continue
        for z in zones:
            zs = str(z)
            if zs and zs not in seen:
                seen.append(zs)
    return seen or None


def _track_context(
    detections: list[dict[str, Any]],
    observed_at: datetime,
    location: Optional[Location],
) -> tuple[dict[str, Any], dict[str, Any], float, Optional[float]]:
    best = max(detections, key=lambda d: float(d.get("score") or 0.0))
    det_scores = [float(d.get("score") or 0.0) for d in detections]
    mean_score = sum(det_scores) / len(det_scores) if det_scores else 0.0
    speed_kmh = (
        float(best["speed_kmh"]) if best.get("speed_kmh") is not None else None
    )
    common = dict(
        schema_version=SCHEMA_VERSION,
        occurred_at=_parse_occurred_at(detections),
        observed_at=observed_at,
        location=location,
        zones=_collect_zones(detections),
    )
    return common, best, mean_score, speed_kmh


def _consolidate_face(cls, track_id, detections, observed_at, location):
    common, best, mean_score, _ = _track_context(detections, observed_at, location)
    return cls(
        **common,
        entity_type="face",
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=[f"track:{track_id}"],
        payload=FacePayload(class_name="face", bbox=best.get("bbox")),
    )


def _consolidate_scene(cls, track_id, detections, observed_at, location):
    common, best, mean_score, _ = _track_context(detections, observed_at, location)
    scene_blob = best.get("scene") if isinstance(best.get("scene"), dict) else {}
    scene_type = (scene_blob or {}).get("type") or best.get("label") or "unknown"
    return cls(
        **common,
        entity_type="scene",
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=[f"track:{track_id}", f"scene:{scene_type}"],
        payload=ScenePayload(
            class_name=str(scene_type),
            scene_type=str(scene_type),
            scene=dict(scene_blob) if scene_blob else None,
            bbox=best.get("bbox"),
        ),
    )


def _consolidate_object(cls, track_id, detections, observed_at, location):
    common, best, mean_score, speed_kmh = _track_context(
        detections, observed_at, location
    )
    class_name, _ = _weighted_vote(detections, "label", "score")
    person_attrs = None
    for d in reversed(detections):
        if isinstance(d.get("person"), dict) and d["person"]:
            person_attrs = dict(d["person"])
            break
    return cls(
        **common,
        entity_type="object",
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=[f"track:{track_id}"],
        payload=ObjectPayload(
            class_name=class_name,
            bbox=best.get("bbox"),
            speed_kmh=speed_kmh,
            person=person_attrs,
        ),
    )


def _consolidate_pose(cls, track_id, detections, observed_at, location):
    common, best, mean_score, _ = _track_context(detections, observed_at, location)
    class_name, _ = _weighted_vote(detections, "label", "score")
    kps = best.get("keypoints") if isinstance(best.get("keypoints"), list) else None
    return cls(
        **common,
        entity_type="pose",
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=[f"track:{track_id}"],
        payload=PosePayload(
            class_name=class_name or "pose",
            bbox=best.get("bbox"),
            keypoints=kps,
        ),
    )


def _consolidate_text(cls, track_id, detections, observed_at, location):
    common, best, mean_score, _ = _track_context(detections, observed_at, location)
    class_name, _ = _weighted_vote(detections, "label", "score")
    text_val = best.get("text")
    cands = [f"track:{track_id}"]
    if text_val:
        cands.insert(0, f"text:{text_val}")
    return cls(
        **common,
        entity_type="text",
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=cands,
        payload=TextPayload(
            class_name=class_name or "text",
            bbox=best.get("bbox"),
            text=str(text_val) if text_val else None,
        ),
    )


def _consolidate_face_id(cls, track_id, detections, observed_at, location):
    common, best, mean_score, _ = _track_context(detections, observed_at, location)
    class_name, _ = _weighted_vote(detections, "label", "score")
    identity_hash = pseudonymize_identity(
        str(best["identity"]) if best.get("identity") else None
    )
    cands = [f"track:{track_id}"]
    if identity_hash:
        cands.insert(0, f"identity:{identity_hash}")
    return cls(
        **common,
        entity_type="face_id",
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=cands,
        payload=IdentityPayload(
            class_name=class_name or "face_id",
            bbox=best.get("bbox"),
            identity=identity_hash,
        ),
    )


def _consolidate_generic(cls, track_id, detections, observed_at, location):
    entity_type = detections[0].get("entity_type") or "object"
    common, best, mean_score, _ = _track_context(detections, observed_at, location)
    class_name, _ = _weighted_vote(detections, "label", "score")
    text_val = best.get("text")
    identity_hash = pseudonymize_identity(
        str(best["identity"]) if best.get("identity") else None
    )
    kps = best.get("keypoints") if isinstance(best.get("keypoints"), list) else None
    cands = [f"track:{track_id}"]
    if identity_hash:
        cands.insert(0, f"identity:{identity_hash}")
    if text_val:
        cands.insert(0, f"text:{text_val}")
    return cls(
        **common,
        entity_type=str(entity_type),
        confidence=max(0.0, min(1.0, mean_score)),
        candidate_ids=cands,
        payload=GenericPayload(
            entity_type=str(entity_type),
            class_name=class_name or str(entity_type),
            bbox=best.get("bbox"),
            text=str(text_val) if text_val else None,
            identity=identity_hash,
            keypoints=kps,
        ),
    )


def _consolidate_vehicle(cls, track_id, detections, observed_at, location):
    common, best, mean_score, speed_kmh = _track_context(
        detections, observed_at, location
    )
    plate_text, plate_vote_conf = _weighted_vote(
        detections, "plate_text", "plate_score"
    )
    if plate_text is None:
        plate_fallback = [
            {**d, "plate_score": d.get("plate_score") or d.get("score", 0.0)}
            for d in detections
        ]
        plate_text, plate_vote_conf = _weighted_vote(
            plate_fallback, "plate_text", "plate_score"
        )

    color, _ = _weighted_vote(detections, "color", "score")
    type_winner, _ = _weighted_vote(detections, "label", "score")

    if plate_text and plate_vote_conf > 0.0:
        confidence = max(0.0, min(1.0, mean_score * (0.5 + 0.5 * plate_vote_conf)))
        plate_confidence = plate_vote_conf
    else:
        confidence = max(0.0, min(1.0, mean_score))
        plate_confidence = None

    candidate_ids = [f"track:{track_id}"]
    if plate_text:
        candidate_ids.insert(0, f"patente:{plate_text}")

    return cls(
        **common,
        entity_type="vehicle",
        confidence=confidence,
        candidate_ids=candidate_ids,
        payload=VehiclePayload(
            color=color,
            vehicle_type=type_winner,
            plate_text=plate_text,
            plate_confidence=plate_confidence,
            bbox=best.get("bbox"),
            speed_kmh=speed_kmh,
        ),
    )


# Despacho por entity_type — agregar consolidator aquí al sumar una capacidad.
CONSOLIDATORS: dict[str, Any] = {
    "face": _consolidate_face,
    "scene": _consolidate_scene,
    "object": _consolidate_object,
    "pose": _consolidate_pose,
    "text": _consolidate_text,
    "face_id": _consolidate_face_id,
    "sign": _consolidate_generic,
    "scene_cls": _consolidate_generic,
    "instance": _consolidate_generic,
    "small_object": _consolidate_generic,
    "anomaly": _consolidate_generic,
    "open_vocab": _consolidate_generic,
    "vehicle": _consolidate_vehicle,
}


def _emit_track(
    cls: type[PerceptionEvent],
    track_id: str,
    detections: list[dict[str, Any]],
    observed_at: datetime,
    location: Optional[Location],
) -> Optional[PerceptionEvent]:
    entity_type = detections[0].get("entity_type") or "vehicle"
    consolidator = CONSOLIDATORS.get(entity_type, _consolidate_vehicle)
    return consolidator(cls, track_id, detections, observed_at, location)


# Re-export útiles para tests / docs de contrato
__all__ = [
    "SCHEMA_VERSION",
    "CONSOLIDATORS",
    "Location",
    "VehiclePayload",
    "ObjectPayload",
    "FacePayload",
    "ScenePayload",
    "PosePayload",
    "TextPayload",
    "IdentityPayload",
    "GenericPayload",
    "PerceptionEvent",
    "pseudonymize_identity",
]
