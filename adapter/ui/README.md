# adapter/ui/

## Para qué sirve

Front del panel: shell HTML + schema AMIS/ECharts + placeholder de preview.

## Cómo funciona

1. `dashboard.html` carga AMIS SDK **local** (`vendor/`) + schema `amis_dashboard.json`.
2. El schema habla con `/media/*`, `/preview.mjpg`, `/events`.
3. Tras clear, el adapter muestra `placeholder_preview.jpg`.
4. Tabla de eventos: filtro por Entidad + columnas Tipo / Color / Detalle
   (scene type · lanes · cruce, attrs persona, texto, identidad, keypoints)
   + pie por `entity_type`.

## Entrada / salida

Estáticos servidos por FastAPI (`STATIC_DIR`). Sin lógica de negocio.
**Sin CDN en runtime** — el panel funciona offline / en edge sin red externa.

## Servicio / deps

Montados en el contenedor `adapter`.

### Vendor (pins)

| Paquete | Versión | Ruta |
|---------|---------|------|
| AMIS SDK | 6.7.0 | `vendor/amis@6.7.0/` (`sdk.css`, `helper.css`, `iconfont.css`, `sdk.js`, fuentes) |
| ECharts | 5.5.1 | `vendor/echarts@5.5.1/echarts.min.js` |

Re-vendor (con red):

```bash
BASE=https://unpkg.com/amis@6.7.0/sdk
DEST=adapter/ui/vendor/amis@6.7.0
mkdir -p "$DEST/thirds/@fortawesome/fontawesome-free/webfonts"
for f in sdk.css helper.css iconfont.css sdk.js \
  iconfont.eot iconfont.svg iconfont.ttf iconfont.woff; do
  curl -fsSL -o "$DEST/$f" "$BASE/$f"
done
FA="$BASE/thirds/@fortawesome/fontawesome-free/webfonts"
for f in fa-brands-400.ttf fa-brands-400.woff2 fa-regular-400.ttf fa-regular-400.woff2 \
  fa-solid-900.ttf fa-solid-900.woff2 fa-v4compatibility.ttf fa-v4compatibility.woff2; do
  curl -fsSL -o "$DEST/thirds/@fortawesome/fontawesome-free/webfonts/$f" "$FA/$f"
done
mkdir -p adapter/ui/vendor/echarts@5.5.1
curl -fsSL -o adapter/ui/vendor/echarts@5.5.1/echarts.min.js \
  https://unpkg.com/echarts@5.5.1/dist/echarts.min.js
```

## Archivos clave

- `dashboard.html` — shell.
- `amis_dashboard.json` — UI declarativa (foto-first).
- `placeholder_preview.jpg` — preview vacío.
- `vendor/` — SDK pinneado (AMIS + ECharts).

## Frontend: AMIS vs Next.js vs SPA (decisión de producto — actualizada)

**Decisión original (superada):** AMIS se queda este sprint y el siguiente.

**Decisión vigente (addendum `addendum-s2-spa-s3`):** esa decisión queda
**superada** — el addendum SPA (Fase 1) reemplaza el plan "AMIS dos sprints
sin rewrite":

- **AMIS sigue siendo el panel en `/`** — no se retira ni se reescribe acá.
  Sigue consumiendo `/events` y mostrando los `entity_type` extendidos.
- **La nueva SPA (Vite, sin Node en el host de runtime) monta en `/app/`**,
  como panel alternativo/nuevo — no como reemplazo inmediato de este panel.
- **AMIS deja de estar en `/` únicamente cuando la SPA cumpla el DoD de su
  Fase 1** (ver diseño/tareas del addendum `addendum-s2-spa-s3`). Hasta ese
  punto, ambos coexisten: `/` = AMIS (este panel), `/app/` = SPA.
- Next.js sigue sin adoptarse — la vía elegida para la nueva UI es la SPA
  Vite del addendum, no Next.js.

## SPA (`/app/`) — Fase 1 implementada (Batch 2, addendum-s2-spa-s3)

- Fuente: `adapter/ui/spa-src/` (Vite + React + TS, sin Next/SSR).
  `npm run build` → `adapter/ui/spa/` (gitignored — ver raíz `.gitignore`;
  se regenera en cada build de imagen, `.gitkeep` solo mantiene el directorio
  en checkouts sin build).
- `adapter/Dockerfile` es multi-stage: stage `node:20-slim` corre
  `npm ci && npm run build` (o `npx tsc -b && npx vite build` en el
  Dockerfile, sin depender del hook `predev`/`prebuild` que asume el layout
  del host); el stage Python final solo copia `adapter/ui/spa/` — **no
  necesita Node en el host de runtime** (verificado con
  `docker compose build adapter` + smoke test `/`, `/app/`, `/health`).
- Montaje en `adapter/app.py`: `StaticFiles(directory=SPA_DIR, html=True)`
  en `/app` — sin fallback adicional a `index.html` en Fase 1 (una sola
  pantalla, sin rutas de cliente propias).
- Tipos: `contracts/epp.gen.ts` (fuente: `adapter/epp_core.py`, ver
  `scripts/gen_epp_types.py` + CI) se copia a
  `spa-src/src/types/epp.gen.ts` (dev: `scripts/copy-types.mjs`; Docker:
  `COPY` directo en el Dockerfile).
- Colores: `scripts/gen_entity_colors.py` extrae por texto los dicts BGR de
  `detection/common/preview.py` (mismo overlay que usa AMIS/MJPEG) y emite
  `spa-src/src/colors/entityColors.gen.ts` en hex RGB — una sola paleta,
  sin duplicar valores a mano. Sin chequeo de CI (a diferencia de
  `epp.gen.ts`); regenerar a mano si `preview.py` cambia de paleta.
- Completitud del análisis: `generation === last_ingest_generation` (ver
  `GET /events`) — no existe un flag `analysis_complete`.

## Qué no es

No es una app React/Vue propia. No contiene detección.
