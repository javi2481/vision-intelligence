# adapter/ui/

## Para qué sirve

Front del panel: shell HTML + schema AMIS/ECharts + placeholder de preview.

## Cómo funciona

1. `dashboard.html` carga AMIS SDK + schema `amis_dashboard.json`.
2. El schema habla con `/media/*`, `/preview.mjpg`, `/events`.
3. Tras clear, el adapter muestra `placeholder_preview.jpg`.
4. Tabla de eventos: filtro por Entidad + columnas Tipo / Color / Detalle
   (scene type · lanes · cruce, attrs persona, texto, identidad, keypoints)
   + pie por `entity_type`.

## Entrada / salida

Estáticos servidos por FastAPI (`STATIC_DIR`). Sin lógica de negocio.

## Servicio / deps

Montados en el contenedor `adapter`. CDN AMIS/ECharts (browser necesita red).

## Archivos clave

- `dashboard.html` — shell.
- `amis_dashboard.json` — UI declarativa (foto-first).
- `placeholder_preview.jpg` — preview vacío.

## Frontend: AMIS vs Next.js (Fase 4)

**Decisión actual:** seguir con AMIS para el panel de percepción.

- Ya consume `/events` y muestra entity_types extendidos.
- Next.js solo tiene sentido como spike **después** de estabilizar backend
  (auth, multi-página, mapas, editor de reglas). No sustituye AMIS hasta
  decisión explícita de producto.

## Qué no es

No es una app React/Vue propia. No contiene detección.
