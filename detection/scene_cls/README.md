# detection/scene_cls/

## Para qué sirve

Clasificación global de imagen (`image_classification` / multilabel): night, rain, etc.

## GATE (Fase 3)

No activar sin: (1) regla AMIS concreta, (2) estimación RAM/latencia, (3) profile `experimental`.

## Servicio

`paddlex-scene-cls` `:8089` — `ENABLE_SCENE_CLS=false` por default.
