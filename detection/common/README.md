# detection/common/

## Para qué sirve

Utilidades compartidas por vehicles/objects/plates y el bridge: tracking IoU,
geometría de frame y overlay de preview.

## Cómo funciona

- `tracking.py` — `IoUTracker` + `iou` (track_id locales, no MOT de modelo).
- `geometry.py` — encode JPEG, resize de inferencia, scale de bboxes.
- `preview.py` — dibuja cajas + labels EN (sin `result.image` chino de PaddleX).

## Entrada / salida

Frames OpenCV BGR / listas de detecciones → JPEG o bboxes reescalados.

## Servicio / deps

Ningún servicio propio. Env: `JPEG_QUALITY`, `BRIDGE_MAX_WIDTH`, `TRACK_IOU_THRESHOLD`.

## Archivos clave

Empezar por el módulo que necesites; el bridge importa `geometry` + `preview`.

## Qué no es

No llama a PaddleX ni al adapter. No contiene lógica de negocio ni reglas.
