# adapter/ui/

## Para qué sirve

Front del panel: shell HTML + schema AMIS/ECharts + placeholder de preview.

## Cómo funciona

1. `dashboard.html` carga AMIS SDK + schema `amis_dashboard.json`.
2. El schema habla con `/media/*`, `/preview.mjpg`, `/events`.
3. Tras clear, el adapter muestra `placeholder_preview.jpg`.

## Entrada / salida

Estáticos servidos por FastAPI (`STATIC_DIR`). Sin lógica de negocio.

## Servicio / deps

Montados en el contenedor `adapter`. CDN AMIS/ECharts (browser necesita red).

## Archivos clave

- `dashboard.html` — shell.
- `amis_dashboard.json` — UI declarativa (foto-first).
- `placeholder_preview.jpg` — preview vacío.

## Qué no es

No es una app React/Vue propia. No contiene detección.
