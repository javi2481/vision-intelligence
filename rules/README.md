# rules/

## Para qué sirve

Sink headless de reglas (perfil Compose `rules`): alerta sobre PerceptionEvents
reenviados por el adapter.

## Cómo funciona

1. Adapter POST a `JETLINKS_WEBHOOK_URL` (apuntar a este servicio).
2. `POST /webhook/events` evalúa regla MVP.
3. Alertas en memoria vía `GET /alerts` (auth `x-api-key`).

Regla MVP: alerta si algún `candidate_ids` empieza con `patente:` **o**
`confidence >= 0.7`.

## Entrada / salida

- **Entrada:** array JSON de eventos (shape permisivo).
- **Salida:** `{received, alerted}` + buffer de alertas.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `rules-sink` (profile `rules`) |
| Puerto | `8850` |
| Env | `RULES_SINK_API_KEY`, `VI_ENV` |

## Archivos clave

- `app.py` — FastAPI completo.

## Qué no es

No reemplaza JetLinks UI (puede coexistir). No importa `epp_core` (modelo propio liviano).
