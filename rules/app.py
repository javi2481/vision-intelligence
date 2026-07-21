"""
rules-sink — capa de reglas headless (perfil Compose `rules`).

Responsabilidad: recibir PerceptionEvent(s) reenviados por el adapter,
evaluar la regla MVP de alerta y exponer alertas para inspección.

NO importa adapter.epp_core (imagen liviana): define su propio modelo
Pydantic permisivo (`extra="ignore"`) con solo los campos que la regla
necesita. La regla vive únicamente en este archivo.

Flujo:
  POST /webhook/events → [PerceptionEvent] → evaluar regla → guardar alerta
  GET  /alerts          → últimas alertas (bounded, en memoria, no persiste)
  GET  /health           → healthcheck sin auth (compose)
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger("rules_sink")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# --- Configuración vía entorno ---
# RULES_SINK_PORT solo lo usa docker-compose para el mapeo de puerto del host;
# el proceso siempre escucha en :8850 dentro del contenedor (ver Dockerfile.rules-sink).
RULES_SINK_API_KEY = os.getenv("RULES_SINK_API_KEY", "demo")
RULES_SINK_MAX_ALERTS = int(os.getenv("RULES_SINK_MAX_ALERTS", "100"))
VI_ENV = os.getenv("VI_ENV", "development").strip().lower()


def _enforce_production_secrets() -> None:
    if VI_ENV != "production":
        return
    key = (RULES_SINK_API_KEY or "").strip()
    if not key or key == "demo":
        raise SystemExit(
            "VI_ENV=production requires RULES_SINK_API_KEY set to a non-demo secret"
        )


_enforce_production_secrets()


class VehiclePayloadIn(BaseModel):
    """Subconjunto permisivo de epp_core.VehiclePayload — solo lo que la regla lee."""

    plate_text: Optional[str] = None

    model_config = {"extra": "ignore"}


class PerceptionEventIn(BaseModel):
    """
    Modelo local del PerceptionEvent de epp-core (schema 1.0).

    Deliberadamente permisivo: solo declara los campos que `evaluate_rule`
    necesita y descarta el resto (`extra="ignore"`) para no acoplarse al
    contrato completo ni importar epp_core.
    """

    candidate_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    payload: VehiclePayloadIn = Field(default_factory=VehiclePayloadIn)

    model_config = {"extra": "ignore"}


def evaluate_rule(event: PerceptionEventIn) -> Optional[str]:
    """
    Regla MVP de alerta.

    Alerta si algún candidate_id es una patente (`patente:...`) o si la
    confianza del evento es alta (>= 0.7). Retorna la razón o None.
    """
    has_plate = any(c.startswith("patente:") for c in event.candidate_ids)
    high_confidence = event.confidence >= 0.7

    if has_plate and high_confidence:
        return "patente+confidence"
    if has_plate:
        return "patente"
    if high_confidence:
        return "confidence"
    return None


# Alertas en memoria, acotadas — se pierden al reiniciar (alcance MVP).
alerts: deque[dict[str, Any]] = deque(maxlen=RULES_SINK_MAX_ALERTS)


def _append_alert(event: PerceptionEventIn, reason: str) -> None:
    alerts.appendleft(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "candidate_ids": event.candidate_ids,
            "confidence": event.confidence,
            "plate_text": event.payload.plate_text,
            "reason": reason,
        }
    )


async def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Valida x-api-key contra RULES_SINK_API_KEY.

    Espejo del contrato del adaptador (`JETLINKS_API_KEY` → header
    `x-api-key`); ambos deben coincidir (default "demo" en ambos lados).
    """
    if x_api_key != RULES_SINK_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing x-api-key",
        )


app = FastAPI(
    title="Vision Intelligence Rules Sink",
    description="Capa de reglas headless — evalúa PerceptionEvents reenviados por el adaptador (EPP v4.6 Punto 12)",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Healthcheck sin auth, usado por Docker Compose."""
    return {"status": "ok"}


@app.post("/webhook/events", dependencies=[Depends(verify_api_key)])
async def webhook_events(events: list[PerceptionEventIn]) -> dict[str, int]:
    """
    Recibe el array de PerceptionEvent que reenvía el adaptador, evalúa la
    regla MVP por evento y guarda una alerta cuando corresponde.
    """
    alerted = 0
    for event in events:
        reason = evaluate_rule(event)
        if reason is not None:
            _append_alert(event, reason)
            alerted += 1
            logger.info(
                "Alert reason=%s candidates=%s conf=%.3f",
                reason,
                event.candidate_ids,
                event.confidence,
            )
    return {"received": len(events), "alerted": alerted}


@app.get("/alerts", dependencies=[Depends(verify_api_key)])
async def get_alerts() -> dict[str, Any]:
    """Últimas alertas (requiere x-api-key), acotadas a RULES_SINK_MAX_ALERTS."""
    items = list(alerts)
    return {"count": len(items), "alerts": items}
