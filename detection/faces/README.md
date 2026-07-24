# detection/faces/

## Para qué sirve

Detectar rostros humanos (bbox + score) vía modelo face PaddleX
(`PP-YOLOE_plus-S_face`) servido como pipeline `object_detection` custom
(`detection/faces/pipeline.yaml`). PaddleX 3.7 no publica pipeline
`face_detection`.

## Cómo funciona

1. Opcional: solo corre si `ENABLE_FACE_DETECTION=true`.
2. El bridge envía el mismo JPEG de inferencia.
3. `infer_faces` → `normalize_face_result` con `track_id` prefijo `f-`.
4. Se concatenan a la lista de detecciones (fallo aislado).

## Entrada / salida

- **Entrada:** `jpeg: bytes`.
- **Salida:** `[{track_id, label:"face", score, bbox, entity_type:"face", frame_ts}]`.
- Fallo HTTP → `None` (no degrada el bridge).

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-faces` (profile `extended`) |
| Puerto | `8083` |
| Env | `PADDLEX_FACES_URL`, `PADDLEX_FACES_PREDICT_PATH` (default `/object-detection`), `ENABLE_FACE_DETECTION` |

## Archivos clave

- `client.py` — `infer_faces`, `normalize_face_result`.
- `pipeline.yaml` — config PaddleX montada en compose.

## Qué no es

No hace reconocimiento facial ni atributos demográficos (eso es `pedestrians/`).
