# bridge/

## Para qué sirve

Orquestador foto-only: elige la foto activa, llama a `detection/*`, empuja
preview e ingest al adapter.

## Cómo funciona

```text
idle ←→ poll /media/current
         ↓ foto
      imread → vehicles∥objects → merge → plates? → overlay → /ingest + /preview/frame
         ↓ clear
       idle
```

`DEMO_MODE=1` emite detecciones sintéticas sin PaddleX.

## Entrada / salida

- **Entrada:** foto en `MEDIA_DIR/images` (vía adapter).
- **Salida:** POST JSON a `/ingest`, JPEG anotado a `/preview/frame`.

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `bridge` (y `bridge-demo`) |
| Depende de | `adapter`, `paddlex*`, paquetes `detection/` |
| Env | `ADAPTER_*`, `PADDLEX_*`, `ENABLE_PLATE_OCR`, `MEDIA_*` |

## Archivos clave

- `main.py` — `run_loop`, `run_detections` (flujo completo).
- `media.py` — resolución de ruta / idle.

## Qué no es

No abre RTSP ni video. No consolida tracks (eso es `adapter/`). No sirve UI.
