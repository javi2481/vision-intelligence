# detection/plates/

## Para qué sirve

OCR opcional de patentes sobre crops de vehículos ya detectados.

## Cómo funciona

1. Gate `ENABLE_PLATE_OCR` (default off).
2. Toma top-K vehículos con `score > OCR_MIN_SCORE`.
3. Recorta bbox en `frame_hires` → JPEG → POST `/ocr`.
4. Filtra textos con regex 5–8 alfanuméricos; merge en `d["plate"]`.
5. Caída/timeout → `plate=None` sin degradar el bridge.

## Entrada / salida

- **Entrada:** crop JPEG o frame+bbox.
- **Salida:** `{text, score}` o `None`.
- Config modelo: `pipeline.yaml` (PP-OCRv5 mobile, compatible con paddle 3.0.0).

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-ocr` |
| Puerto | `8081` |
| Env | `ENABLE_PLATE_OCR`, `PADDLEX_OCR_URL`, `OCR_*` |

## Archivos clave

- `client.py` — `enrich_vehicles_with_plates`, `infer_plate_ocr`.
- `pipeline.yaml` — config servida en la imagen PaddleX.

## Qué no es

No es un LPR dedicado de producción; OCR genérico + regex. No detecta vehículos.
