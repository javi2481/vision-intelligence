"""
epp-core draft contract (schema_version=\"1.0-draft\").

Contrato sagrado del Producto B — Vision Intelligence (EPP v4.6, Punto 12).
Única pieza portable: entra un dict (detección PaddleX), sale un PerceptionEvent.
Sin lógica de reglas de negocio; solo normalización y consolidación de tracks.

Asumción del JSON PaddleX PP-Vehicle (por detección)::

    {
      \"track_id\": \"42\",           # o int; se normaliza a str
      \"label\": \"car\",             # vehicle_type
      \"score\": 0.91,              # confianza de detección [0,1]
      \"bbox\": [x1, y1, x2, y2],
      \"color\": \"white\",           # atributo de vehículo (opcional)
      \"plate\": {                  # lectura de patente (opcional)
        \"text\": \"ABC123\",
        \"score\": 0.87
      },
      \"speed_kmh\": 45.2,          # opcional (radar / estimación)
      \"frame_ts\": \"2026-07-18T15:00:00.123Z\"  # opcional ISO-8601
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
    """Payload tipado del vehículo consolidado (pistas, no veredictos)."""

    color: Optional[str] = None
    vehicle_type: Optional[str] = None
    plate_text: Optional[str] = None
    plate_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bbox: Optional[list[float]] = None
    speed_kmh: Optional[float] = None


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
    """Construye un PerceptionEvent a partir de las detecciones de un track."""
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

    # bbox / speed del frame con mayor score
    best = max(detections, key=lambda d: float(d.get("score") or 0.0))
    det_scores = [float(d.get("score") or 0.0) for d in detections]
    mean_score = sum(det_scores) / len(det_scores) if det_scores else 0.0

    # Confianza comparable: media de detección * peso de voto de patente
    # (si no hay patente, solo media de detección).
    if plate_text and plate_vote_conf > 0.0:
        confidence = max(0.0, min(1.0, mean_score * (0.5 + 0.5 * plate_vote_conf)))
        plate_confidence = plate_vote_conf
    else:
        confidence = max(0.0, min(1.0, mean_score))
        plate_confidence = None

    candidate_ids: list[str] = [f"track:{track_id}"]
    if plate_text:
        candidate_ids.insert(0, f"patente:{plate_text}")

    payload = VehiclePayload(
        color=color,
        vehicle_type=type_winner,
        plate_text=plate_text,
        plate_confidence=plate_confidence,
        bbox=best.get("bbox"),
        speed_kmh=(
            float(best["speed_kmh"]) if best.get("speed_kmh") is not None else None
        ),
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
