# detection/face_id/

## Para qué sirve

Identidad facial vía PaddleX `face_recognition` (match / label).
El bbox de rostro “anónimo” sigue en `faces/`.

## Cómo funciona

1. `ENABLE_FACE_ID=true` + `paddlex-face-id` (profile `extended`).
2. Emite `entity_type:"face_id"` con `identity` + score.

## Gate / aviso

Requiere galería/base de identidades configurada en el pipeline PaddleX.
Sin eso, labels suelen ser `unknown`. Evaluar aspectos legales/privacidad.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-face-id` `:8087` |
| Env | `ENABLE_FACE_ID`, `PADDLEX_FACE_ID_URL` |

## Qué no es

No sustituye `detection/faces/` (solo detección).
