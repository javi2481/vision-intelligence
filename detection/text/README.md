# detection/text/

## Para qué sirve

OCR de carteles / texto en escena (no filtrado a patentes). Reusa
`paddlex-ocr` (:8081).

## Cómo funciona

1. `ENABLE_SCENE_OCR=true`.
2. POST del JPEG completo a `/ocr`.
3. Emite dets `entity_type:"text"` con `text` + score (top-K por score).

## Entrada / salida

- **Entrada:** `jpeg: bytes`.
- **Salida:** `[{track_id:t-*, label:"text", text, score, bbox, entity_type:"text"}]`.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | reusa `paddlex-ocr` |
| Env | `ENABLE_SCENE_OCR`, `SCENE_OCR_MIN_SCORE`, `SCENE_OCR_MAX_LINES`, `PADDLEX_OCR_URL` |

## Archivos clave

- `client.py`

## Qué no es

No reemplaza `plates/` (regex de patente sobre crops de vehículos).
