# detection/signs/

## Para qué sirve

Señales de tránsito vía `object_detection` (COCO filtrado o fine-tune propio).

## Cómo funciona

1. `ENABLE_SIGNS=true` + `paddlex-signs` (profile `extended`).
2. Filtra labels en `SIGNS_LABELS` (env CSV o default COCO señales).
3. Emite `entity_type:"sign"` con track `s-*`.

## Fine-tune

Apuntar `VI_PIPELINE` / pesos del servicio a un object_detection entrenado
con clases propias (speed_limit, yield, …) y actualizar `SIGNS_LABELS`.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-signs` `:8088` |
| Env | `ENABLE_SIGNS`, `PADDLEX_SIGNS_URL`, `SIGNS_LABELS` |

## Qué no es

No reemplaza `objects/` COCO general (puede solaparse con traffic light/stop sign).
