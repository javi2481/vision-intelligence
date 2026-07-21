# detection/vehicles/

## Para qué sirve

Detectar vehículos y leer atributos (tipo + color) vía PaddleX
`vehicle_attribute_recognition`.

## Cómo funciona

1. El bridge envía un JPEG de inferencia.
2. `infer_vehicles` hace POST a `PADDLEX_URL` + `PADDLEX_PREDICT_PATH`.
3. `normalize_vehicle_result` traduce la respuesta a dicts del adapter.
4. Asigna `track_id` con prefijo `v-` (IoU tracker).
5. `plate` queda `None` hasta que `detection/plates` lo complete (si OCR on).

## Entrada / salida

- **Entrada:** `jpeg: bytes`.
- **Salida:** `[{track_id, label, score, color, bbox, plate, frame_ts, entity_type:"vehicle"}]`.
- Fallo HTTP → `None` (el bridge marca `degraded`).

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex` |
| Puerto | `8080` |
| Env | `PADDLEX_URL`, `PADDLEX_PREDICT_PATH`, `HTTP_TIMEOUT` |

## Archivos clave

- `client.py` — mirar primero (`infer_vehicles`, `normalize_vehicle_result`).

## Qué no es

No dibuja preview ni corre OCR. No es object detection general (eso es `objects/`).
