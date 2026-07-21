# adapter/

## Para qué sirve

API FastAPI: media (upload/select/clear), ingest de detecciones, consolidación
a `PerceptionEvent`, preview MJPEG y shell del dashboard.

## Cómo funciona

1. Watcher / upload selecciona foto → bridge la consume.
2. `POST /ingest` acumula por `track_id` (TTL); con foto activa finaliza al instante.
3. `epp_core.PerceptionEvent.consolidate_and_emit` → buffer `/events`.
4. UI en `ui/` se sirve en `/` y `/static/…`.
5. Opcional: forward a JetLinks / rules-sink.

## Entrada / salida

- **Entrada:** detecciones JSON del bridge; multipart upload de fotos.
- **Salida:** `PerceptionEvent` en `/events`; preview JPEG/MJPEG.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `adapter` |
| Puerto | `8000` |
| Env | `STATIC_DIR=/app/adapter/ui`, `MEDIA_DIR`, `TRACK_TTL_*`, `JETLINKS_*` |

## Archivos clave

- `app.py` — endpoints.
- `epp_core.py` — contrato portable.
- `ui/` — AMIS + placeholder.

## Qué no es

No corre PaddleX. No decide reglas de alerta (ver `rules/`).
