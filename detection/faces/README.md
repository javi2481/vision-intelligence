# detection/faces/

## Para qué sirve

Detectar rostros humanos (bbox + score) vía PaddleX `face_detection`.

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
| Env | `PADDLEX_FACES_URL`, `ENABLE_FACE_DETECTION` |

## Archivos clave

- `client.py` — `infer_faces`, `normalize_face_result`.

## Qué no es

No hace reconocimiento facial ni atributos demográficos (eso es `pedestrians/`).
