# detection/objects/

## Para qué sirve

Detección general COCO (~80 clases: person, dog, bottle, traffic light, …)
vía PaddleX `object_detection`.

## Cómo funciona

1. Corre en paralelo al pipeline de vehículos (mismo JPEG).
2. `infer_objects` → respuesta cruda.
3. `attach_object_track_ids` asigna `track_id` con prefijo `o-`.
4. `merge_coco_detections` descarta cajas COCO de clase vehículo ya cubiertas
   por `vehicles/` (ese pipeline trae color/plate). Labels como `person` se
   conservan siempre.

## Entrada / salida

- **Entrada:** `jpeg: bytes` (+ lista vehicle dets para el merge).
- **Salida:** `[{track_id, label, score, bbox, entity_type:"object", frame_ts}]`.
- Fallo HTTP → `None` (aislado: **no** degrada el bridge).

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-objects` |
| Puerto | `8082` |
| Env | `PADDLEX_OBJECTS_URL`, `PADDLEX_OBJECTS_PREDICT_PATH` |

## Archivos clave

- `client.py` — `infer_objects`, `merge_coco_detections`, `normalize_object_detection_result`.

## Qué no es

**No hay detector de personas dedicado.** `person` es una clase COCO dentro de
este módulo. No inventar carpeta `persons/`.
