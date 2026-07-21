"""
epp-core draft contract (schema_version=\"1.0-draft\").

Ubicación: adapter/epp_core.py — contrato portable del Producto B
(Vision Intelligence, EPP v4.6 Punto 12).

Única pieza portable: entra un dict (detección del bridge), sale un PerceptionEvent.
Sin lógica de reglas de negocio; solo normalización y consolidación de tracks.

Asumción del JSON de detección (por ítem)::

    {
      \"track_id\": \"v-42\",         # o \"o-1\" / \"f-1\" / \"scene-0\"
      \"label\": \"car\",             # vehicle_type, class_name COCO, face, scene_type
      \"score\": 0.91,
      \"bbox\": [x1, y1, x2, y2],
      \"color\": \"white\",           # opcional (vehicles)
      \"plate\": {\"text\": \"ABC123\", \"score\": 0.87},  # opcional
      \"entity_type\": \"vehicle\",   # vehicle | object | face | scene
      \"person\": {...},             # opcional (attrs pedestrians sobre person)
      \"scene\": {...},              # opcional (entity_type scene)
      \"frame_ts\": \"2026-07-18T15:00:00.123Z\"
    }

Variantes aceptadas: plate_text/plate_score en raíz, conf/confidence
en lugar de score, boxes en lugar de bbox.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    """Payload tipado de la entidad consolidada (pistas, no veredictos).

    Nombre histórico ("Vehicle...") preservado para no romper referencias en
    adapter/app.py y el dashboard; el field set cubre también object/face/scene.
    """

    color: Optional[str] = None
    vehicle_type: Optional[str] = None
    class_name: Optional[str] = None
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None
    speed_kmh: Optional[float] = None
    person: Optional[dict[str, Any]] = None
    scene_type: Optional[str] = None
    scene: Optional[dict[str, Any]] = None
    text: Optional[str] = None
    identity: Optional[str] = None
    keypoints: Optional[list[Any]] = None


class PerceptionEvent(BaseModel):
    """
    Sobre común epp-core (borrador).

    Garantías:
      #1 Dos tiempos: occurred_at (frame) y observed_at (procesamiento).
      #3 Confianza comparable en [0.0, 1.0].
      #4 Pistas, no veredictos: candidate_ids.
      #6 Versionado: schema_version=\"1.0-draft\".
    """

    schema_version: str = "1.0-draft"
    entity_type: str = "vehicle"
    occurred_at: datetime
    observed_at: datetime = Field(default_factory=_utc_now)
    confidence: float = Field(..., ge=0.0, le=1.0)
    candidate_ids: list[str] = Field(default_factory=list)
    location: Optional[Location] = None
    payload: VehiclePayload

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
    ) -> list["PerceptionEvent"]:
        """
        Agrupa detecciones por track_id y aplica votación temporal
        ponderada por confianza (score) para plate_text y color.

        Emite un PerceptionEvent por track consolidado.
        """
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
        # Default "vehicle" preserva compat con detecciones sintéticas de
        # bridge-demo (no traen entity_type).
        "entity_type": raw.get("entity_type") or "vehicle",
        "person": raw.get("person") if isinstance(raw.get("person"), dict) else None,
        "scene": raw.get("scene") if isinstance(raw.get("scene"), dict) else None,
        "text": raw.get("text"),
        "identity": raw.get("identity"),
        "keypoints": raw.get("keypoints") if isinstance(raw.get("keypoints"), list) else None,
    }


def _weighted_vote(
    detections: list[dict[str, Any]],
    value_key: str,
    weight_key: str,
) -> tuple[Optional[str], float]:
    """
    Votación por mayoría ponderada por score.

    Retorna (valor_ganador, peso_normalizado_del_ganador).
    """
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
    """Usa el frame_ts más reciente del track; fallback a now()."""
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


def _emit_track(
    cls: type[PerceptionEvent],
    track_id: str,
    detections: list[dict[str, Any]],
    observed_at: datetime,
    location: Optional[Location],
) -> Optional[PerceptionEvent]:
    """Construye un PerceptionEvent a partir de las detecciones de un track.

    `entity_type` se deriva de las detecciones del track (primera detección
    no-nula; en la práctica todo un track comparte entity_type, ya que
    vehicle y object_detection usan trackers e IDs con prefijo separados
    "v-"/"o-" en detection/vehicles y detection/objects). Para `entity_type == "object"` se vota
    `class_name` (la label COCO) en vez de `color`/`plate_text`, y se omite
    el boost de confianza por patente (concepto propio de vehículos).
    """
    entity_type = detections[0].get("entity_type") or "vehicle"

    # bbox / speed del frame con mayor score (común a ambas ramas)
    best = max(detections, key=lambda d: float(d.get("score") or 0.0))
    det_scores = [float(d.get("score") or 0.0) for d in detections]
    mean_score = sum(det_scores) / len(det_scores) if det_scores else 0.0
    speed_kmh = (
        float(best["speed_kmh"]) if best.get("speed_kmh") is not None else None
    )

    if entity_type == "face":
        confidence = max(0.0, min(1.0, mean_score))
        return cls(
            schema_version="1.0-draft",
            entity_type="face",
            occurred_at=_parse_occurred_at(detections),
            observed_at=observed_at,
            confidence=confidence,
            candidate_ids=[f"track:{track_id}"],
            location=location,
            payload=VehiclePayload(
                class_name="face",
                bbox=best.get("bbox"),
            ),
        )

    if entity_type == "scene":
        scene_blob = best.get("scene") if isinstance(best.get("scene"), dict) else {}
        scene_type = (
            (scene_blob or {}).get("type")
            or best.get("label")
            or "unknown"
        )
        confidence = max(0.0, min(1.0, mean_score))
        return cls(
            schema_version="1.0-draft",
            entity_type="scene",
            occurred_at=_parse_occurred_at(detections),
            observed_at=observed_at,
            confidence=confidence,
            candidate_ids=[f"track:{track_id}", f"scene:{scene_type}"],
            location=location,
            payload=VehiclePayload(
                class_name=str(scene_type),
                scene_type=str(scene_type),
                scene=dict(scene_blob) if scene_blob else None,
                bbox=best.get("bbox"),
            ),
        )

    if entity_type == "object":
        class_name, _ = _weighted_vote(detections, "label", "score")
        confidence = max(0.0, min(1.0, mean_score))
        candidate_ids: list[str] = [f"track:{track_id}"]
        person_attrs = None
        for d in reversed(detections):
            if isinstance(d.get("person"), dict) and d["person"]:
                person_attrs = dict(d["person"])
                break

        payload = VehiclePayload(
            class_name=class_name,
            bbox=best.get("bbox"),
            speed_kmh=speed_kmh,
            person=person_attrs,
        )

        return cls(
            schema_version="1.0-draft",
            entity_type="object",
            occurred_at=_parse_occurred_at(detections),
            observed_at=observed_at,
            confidence=confidence,
            candidate_ids=candidate_ids,
            location=location,
            payload=payload,
        )

    # Tipos extendidos / experimentales: pose, text, face_id, sign, …
    _GENERIC_TYPES = {
        "pose",
        "text",
        "face_id",
        "sign",
        "scene_cls",
        "instance",
        "small_object",
        "anomaly",
        "open_vocab",
    }
    if entity_type in _GENERIC_TYPES:
        class_name, _ = _weighted_vote(detections, "label", "score")
        confidence = max(0.0, min(1.0, mean_score))
        text_val = best.get("text")
        identity_val = best.get("identity")
        kps = best.get("keypoints") if isinstance(best.get("keypoints"), list) else None
        cands = [f"track:{track_id}"]
        if identity_val:
            cands.insert(0, f"identity:{identity_val}")
        if text_val:
            cands.insert(0, f"text:{text_val}")
        return cls(
            schema_version="1.0-draft",
            entity_type=str(entity_type),
            occurred_at=_parse_occurred_at(detections),
            observed_at=observed_at,
            confidence=confidence,
            candidate_ids=cands,
            location=location,
            payload=VehiclePayload(
                class_name=class_name or str(entity_type),
                bbox=best.get("bbox"),
                text=str(text_val) if text_val else None,
                identity=str(identity_val) if identity_val else None,
                keypoints=kps,
            ),
        )

    # entity_type == "vehicle": comportamiento sin cambios.
    plate_text, plate_vote_conf = _weighted_vote(
        detections, "plate_text", "plate_score"
    )
    # plate_score puede ser None; usar score de detección como peso
    if plate_text is None:
        plate_fallback = [
            {**d, "plate_score": d.get("plate_score") or d.get("score", 0.0)}
            for d in detections
        ]
        plate_text, plate_vote_conf = _weighted_vote(
            plate_fallback, "plate_text", "plate_score"
        )

    color, _ = _weighted_vote(detections, "color", "score")

    # vehicle_type: modo (mayor score acumulado por label)
    type_winner, _ = _weighted_vote(detections, "label", "score")

    # Confianza comparable: media de detección * peso de voto de patente
    # (si no hay patente, solo media de detección).
    if plate_text and plate_vote_conf > 0.0:
        confidence = max(0.0, min(1.0, mean_score * (0.5 + 0.5 * plate_vote_conf)))
        plate_confidence = plate_vote_conf
    else:
        confidence = max(0.0, min(1.0, mean_score))
        plate_confidence = None

    candidate_ids = [f"track:{track_id}"]
    if plate_text:
        candidate_ids.insert(0, f"patente:{plate_text}")

    payload = VehiclePayload(
        color=color,
        vehicle_type=type_winner,
        plate_text=plate_text,
        plate_confidence=plate_confidence,
        bbox=best.get("bbox"),
        speed_kmh=speed_kmh,
    )

    return cls(
        schema_version="1.0-draft",
        entity_type="vehicle",
        occurred_at=_parse_occurred_at(detections),
        observed_at=observed_at,
        confidence=confidence,
        candidate_ids=candidate_ids,
        location=location,
        payload=payload,
    )
