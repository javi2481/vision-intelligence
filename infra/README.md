# infra/

## Para qué sirve

Imagen Docker compartida de PaddleX y su entrypoint. Los tres servicios
(`paddlex`, `paddlex-ocr`, `paddlex-objects`) usan la misma build y se
diferencian por `VI_PIPELINE` / `VI_PORT`.

## Cómo funciona

1. `Dockerfile.paddlex` instala `paddlex[cv,serving,ocr]`.
2. Copia `detection/plates/pipeline.yaml` como OCR v5 mobile.
3. `entrypoint.paddlex.sh` ejecuta `paddlex --serve --pipeline … --port …`.

## Entrada / salida

HTTP serving de cada pipeline (paths `/vehicle-attribute-recognition`,
`/object-detection`, `/ocr` según el servicio).

## Servicio / deps

Build: `infra/Dockerfile.paddlex`. Env: `VI_PIPELINE`, `VI_PORT`, `VI_DEVICE`, `VI_USE_HPIP`.

## Archivos clave

- `Dockerfile.paddlex`
- `entrypoint.paddlex.sh`

## Qué no es

No contiene lógica de producto Python. No es el bridge ni el adapter.
