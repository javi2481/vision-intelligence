# detection/pose/

## Para qué sirve

Keypoints / pose humana vía PaddleX `human_keypoint_detection`.

## Cómo funciona

1. `ENABLE_POSE=true` + servicio `paddlex-pose` (profile `extended`).
2. `infer_pose` → dets `entity_type:"pose"` con `keypoints` y track `k-*`.

## Entrada / salida

- **Entrada:** `jpeg: bytes`.
- **Salida:** `[{track_id, label:"person_pose", score, bbox, keypoints, entity_type:"pose"}]`.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-pose` `:8086` (profile `extended`) |
| Env | `ENABLE_POSE`, `PADDLEX_POSE_URL` |

## Archivos clave

- `client.py`

## Qué no es

No reemplaza `objects/` person ni `pedestrians/` attrs.
