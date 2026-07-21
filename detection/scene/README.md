# detection/scene/

## Para qué sirve

Escena vial vía PaddleX `semantic_segmentation`: `scene_type`, ratios,
carriles y cruces peatonales (según pesos / modo).

## Cómo funciona

1. `ENABLE_SCENE_SEG=true` + profile `extended`.
2. `infer_scene` → `labelMap` → ratios / heurística.
3. Emite det `scene-0`.
4. Modos `SCENE_LABEL_MODE`:
   - `cityscapes` — calle/autopista (default)
   - `lane` — 4 clases PP-Vehicle → `scene.lanes`
   - `bdd_marks` — BDD lane marks → `scene.lanes` + `scene.crosswalk`

## Entrada / salida

- **Entrada:** `jpeg: bytes`.
- **Salida:** `entity_type:"scene"` con `scene:{type,ratios,infra,lanes,crosswalk}`.
- Fallo HTTP → `None`.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-scene` (profile `extended`) |
| Puerto | `8085` |
| Env | `ENABLE_SCENE_SEG`, `SCENE_LABEL_MODE`, `SCENE_PIPELINE_CONFIG`, `CROSSWALK_MIN_RATIO` |

## Archivos clave

- `client.py`
- `pipeline_lane.yaml` — stub pesos lane (montar `model_dir`)
- `pipeline_bdd_marks.yaml` — stub fine-tune crosswalk

## Qué no es

Cityscapes default no trae carriles/cruces. Sin `model_dir` los YAML stub
no segmentan marcas; hay que descargar/entrenar pesos.
