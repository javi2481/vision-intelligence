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

## Frontend: AMIS vs Next.js

**Decisión:** AMIS se queda este sprint y el siguiente.

- Ya consume `/events` y muestra entity_types extendidos.
- Next.js solo tiene sentido como spike **después** de estabilizar backend
  (auth, multi-página, mapas, editor de reglas). No sustituye AMIS hasta
  decisión explícita de producto.

## Qué no es

No es una app React/Vue propia. No contiene detección.
