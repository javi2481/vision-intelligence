# detection/

## Para qué sirve

Código cliente de las capacidades de IA (HTTP + normalize). Bridge orquesta.

## Cómo funciona

```text
bridge → vehicles + objects (+ ENABLE_* extended/experimental)
       → merge → plates? → ingest
```

## Capacidades

| Carpeta | Capacidad | Profile / flag |
|---------|-----------|----------------|
| [vehicles/](vehicles/) | Tipo/color vehículo | default |
| [objects/](objects/) | COCO (incl. person) | default |
| [plates/](plates/) | OCR patente | `ENABLE_PLATE_OCR` |
| [faces/](faces/) | Rostros | extended / `ENABLE_FACE_DETECTION` |
| [pedestrians/](pedestrians/) | Attrs persona | extended / `ENABLE_PEDESTRIAN_ATTRS` |
| [scene/](scene/) | Escena / lanes / crosswalk | extended / `ENABLE_SCENE_SEG` |
| [pose/](pose/) | Keypoints | extended / `ENABLE_POSE` |
| [text/](text/) | OCR carteles | `ENABLE_SCENE_OCR` (reusa ocr) |
| [face_id/](face_id/) | Identidad facial | extended / `ENABLE_FACE_ID` |
| [signs/](signs/) | Señales | extended / `ENABLE_SIGNS` |
| [scene_cls/](scene_cls/) | Clasif. escena | experimental + GATE |
| [instances/](instances/) | Instance seg | experimental + GATE |
| [small_objects/](small_objects/) | Small objects | experimental + GATE |
| [anomaly/](anomaly/) | Anomaly | experimental + GATE |
| [open_vocab/](open_vocab/) | Open-vocab | experimental + GATE |
| [common/](common/) | Tracker, geometry, preview | — |

## Qué no es

No levanta modelos ni Docker: solo clientes HTTP + normalización.
