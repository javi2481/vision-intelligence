# Vision Intelligence — Producto B (Sprint 1)

Orquestar, no inventar. Pipeline Docker-first: **Webcam → MediaMTX → PaddleX vehicle_attribute_recognition → Adaptador epp-core → AMIS/ECharts** (+ JetLinks opcional).

Filosofía EPP v4.6 Punto 12: cero código de IA propio. Solo traducción JSON, consolidación de tracks y configuración.

## Arquitectura

```text
[Host FFmpeg] --RTSP--> [MediaMTX] --RTSP--> [Bridge 1 FPS]
                                              |
                                              v
                    [PaddleX vehicle_attribute_recognition]  (CPU o GPU)
                                              |
                                         JSON detecciones
                                              v
                              [Adapter FastAPI]  track_cache + votación
                                              |
                         PerceptionEvent (epp-core 1.0-draft)
                         /         |          \
                      buffer    webhook     static
                         |         |          |
                      AMIS+ECharts JetLinks   dashboard.html
```

Separación estricta:

| Capa | Responsabilidad |
|------|-----------------|
| FastAPI (`adapter`) | Normaliza y consolida tracks |
| JetLinks (opcional) | Decide reglas (SQL visual) |
| AMIS + ECharts | Muestra |

### Mejora de ingesta (vs. JPEG directo al adaptador)

El bridge envía frames a **PaddleX** y solo JSON al adaptador. El adaptador permanece portable a edge (RK3588): entra `dict`, sale `PerceptionEvent`.

## Bridge: FPS, ancho de inferencia y medición

### Variables de entorno del bridge

| Variable | Default | Propósito |
|----------|---------|-----------|
| `BRIDGE_FPS` | `2` | Frecuencia de **inferencia** PaddleX (detección). Independiente del preview. |
| `PREVIEW_FPS` | `15` | Frecuencia del video en el panel. Cada frame se muestra con los últimos recuadros; la detección corre en paralelo. |
| `BRIDGE_MAX_WIDTH` | `1280` | Ancho máximo de la imagen que se envía a inferencia. Sobre este umbral se reduce solo la copia de inferencia (ver abajo); a la par o por debajo no hay resize. |
| `BRIDGE_METRICS_EVERY` | `30` | Cada cuántos frames inferidos se emite una línea de métricas en el log. |

### OCR de patente (servicio `paddlex-ocr`, opcional)

Segundo servicio PaddleX, misma imagen (`Dockerfile.paddlex`), pipeline `OCR` en vez de `vehicle_attribute_recognition`, puerto propio (`8081` por default). `entrypoint.paddlex.sh` selecciona el pipeline vía `VI_PIPELINE`/`VI_PORT`: sin esas variables, sirve el pipeline de atributos como hoy (default preservado).

> **Nota de compatibilidad**: `paddlex-ocr` usa un config propio (`paddlex_ocr_pipeline.yaml`, copiado a `/opt/paddlex/pipelines/ocr_v5_mobile.yaml` en la imagen, seleccionado vía `VI_PIPELINE=${OCR_PIPELINE_CONFIG}`) que pinea el pipeline OCR a **PP-OCRv5 mobile** (det+rec) en vez de los modelos PP-OCRv6 que trae por default `paddlex==3.7.2`. PP-OCRv6 no es compatible con el motor de inferencia de la imagen base `paddlepaddle/paddle:3.0.0` (`PADDLE_BASE_IMAGE`) — falla con `ValueError: Type of attribute: strides is not right` (mismatch de formato de modelo/PIR) al crear el predictor. El servicio `paddlex` (atributos) **no** se toca: sigue con `vehicle_attribute_recognition` y la misma `PADDLE_BASE_IMAGE`, para no arriesgar ese pipeline. Si en el futuro se sube `PADDLE_BASE_IMAGE` a una versión con soporte PP-OCRv6, se puede volver a `VI_PIPELINE=OCR` (o `OCR_PIPELINE_CONFIG` vacío) sin tocar código.

En `rtsp_bridge.py`, después de `_scale_detections(...)` (bbox ya en coords `frame_hires`) y antes del único `_post_json`, si `ENABLE_PLATE_OCR=true`: cada `OCR_EVERY_N_FRAMES` frames, toma hasta `OCR_TOPK` detecciones con `score > OCR_MIN_SCORE` (orden descendente), recorta `frame_hires` en el bbox (clip a bordes + guard de crop degenerado), envía el JPEG a `POST {PADDLEX_OCR_URL}/ocr`, filtra `rec_texts`/`rec_scores` con una regex de patente (5-8 alfanuméricos) y hace merge del texto de mayor score en `d["plate"] = {"text", "score"}` — mismo shape que `_demo_detections`. Sin match o servicio caído/timeout → `plate=None` para esa detección, **sin** marcar el bridge degradado globalmente (`_notify_degraded` queda reservado al pipeline attr primario).

| Variable | Default | Propósito |
|----------|---------|-----------|
| `ENABLE_PLATE_OCR` | `false` | Gate general: en `false` no se hace ningún crop/POST/OCR (cero costo extra). |
| `PADDLEX_OCR_URL` | `http://paddlex-ocr:8081` | Base URL del servicio OCR. |
| `OCR_MIN_SCORE` | `0.7` | Score mínimo de detección para intentar OCR. |
| `OCR_EVERY_N_FRAMES` | `5` | Cada cuántos frames (con OCR habilitado) se ejecuta el bloque OCR. |
| `OCR_TOPK` | `3` | Máximo de detecciones OCR-eadas por frame elegible (evita apilar llamadas secuenciales en CPU con muchos vehículos). |
| `OCR_HTTP_TIMEOUT` | `5` | Timeout HTTP (segundos) propio del POST a `paddlex-ocr`, independiente de `HTTP_TIMEOUT`. |

**Presupuesto CPU/FPS**: OCR es una segunda inferencia PaddleX secuencial por detección elegible — con `OCR_TOPK=3` y una escena con varios vehículos por encima de `OCR_MIN_SCORE`, el frame puede tardar sensiblemente más que sin OCR en CPU. Ajustar `OCR_EVERY_N_FRAMES` (menos frecuente) y `OCR_TOPK` (menos detecciones por frame) para mantener el FPS efectivo (ver métricas de log) dentro de lo esperado. Medir con `docker stats vi-paddlex-ocr` y la métrica `effective_fps` del bridge antes/después de habilitar.

**Limitación**: el pipeline `OCR` de PaddleX es un OCR genérico (no un modelo de reconocimiento de patentes dedicado). La regex de 5-8 alfanuméricos filtra ruido, pero puede producir falsos positivos/negativos con placas fuera de ese formato o con texto ambiguo en la escena (carteles, logos). Para producción se recomienda evaluar un modelo/pipeline específico de patentes.

## Selector de muestra local + preview anotado (overlay-preview)

Además de RTSP en vivo, el stack puede correr contra un video o foto de muestra
local, con overlay de bbox+patente dibujado por el bridge y expuesto como
preview en el dashboard — sin FFmpeg/MediaMTX en el camino, sin tocar
`epp_core.py`, sin React ni webcam del usuario (EPP Punto 12).

**Flujo**: AMIS (`select`) → `POST /media/select` (adapter valida contra el
allow-list real de `videos_muestra/`/`imagenes_muestra/`) → el bridge polea
`GET /media/current` y reabre su `cv2.VideoCapture` en caliente (sin reiniciar
el contenedor) → dibuja bbox+label sobre `frame_hires` → empuja el JPEG
anotado a `POST /preview/frame` → el dashboard lo muestra vía `/preview.mjpg`
(video) o `/preview.jpg` (foto). `POST /ingest` sigue igual: `/events` no se
ve afectado (dual output).

### Variables de entorno

| Variable | Default | Propósito |
|----------|---------|-----------|
| `MEDIA_DIR` | `/media` | Punto de montaje RO común (adapter+bridge): `{MEDIA_DIR}/videos` ← `./videos_muestra`, `{MEDIA_DIR}/images` ← `./imagenes_muestra`. |
| `MEDIA_POLL_INTERVAL` | `1.0` | Segundos entre polls del bridge a `GET /media/current` para detectar hot-swap. |
| `SOURCE_URL` | (vacío → `RTSP_URL`) | Generaliza el origen del bridge: URL RTSP o ruta local de archivo. Solo aplica mientras no haya una muestra seleccionada. |
| `ADAPTER_MEDIA_CURRENT_URL` | `http://adapter:8000/media/current` | URL que el bridge polea para hot-swap. |
| `ADAPTER_PREVIEW_FRAME_URL` | `http://adapter:8000/preview/frame` | URL donde el bridge empuja el JPEG anotado. |
| `PREVIEW_MJPEG_INTERVAL` | `0.05` | (adapter) Intervalo de re-emisión de `/preview.mjpg`. |

### Endpoints nuevos (adapter)

| Endpoint | Descripción |
|----------|-------------|
| `GET /media/list` | `{items:[{name,type}]}` — allow-list real (archivos físicamente presentes, no recursivo). |
| `GET /media/current` | `{name,type,generation}` — fuente activa. |
| `POST /media/select {name}` | Selecciona una muestra allow-listed; `400` si `name` no está en la lista (path traversal incluido); no cambia la fuente activa. |
| `POST /preview/frame` | (uso interno del bridge) Body `image/jpeg` crudo → guarda el último frame anotado. |
| `GET /preview.mjpg` | Stream `multipart/x-mixed-replace` del preview anotado (video). |
| `GET /preview.jpg` | Único JPEG anotado más reciente (foto); `503` si aún no hay frame. |

### Escenarios de smoke manual (ejecutar tras `docker compose up --build`)

1. **Boxes alineados en `Brasil6.mp4`**: en el dashboard, seleccionar `Brasil6.mp4` en "Fuente local (muestras)"; el preview debe mostrar rectángulos+etiqueta alineados sobre los vehículos detectados.
2. **Foto sirve un solo frame anotado**: seleccionar una muestra de imagen (`imagenes_muestra/sample_*.jpg`); `/preview.jpg` debe devolver un único JPEG anotado (o el frame se mantiene fijo en `/preview.mjpg` si el navegador cae al fallback `onerror`).
3. **`/events` sigue poblándose durante el preview**: con una muestra de video seleccionada, la tabla de eventos y las 4 tarjetas KPI deben seguir actualizándose exactamente igual que antes de este cambio.
4. **Selección fuera del allow-list se rechaza**: `curl -X POST http://localhost:8000/media/select -H "Content-Type: application/json" -d "{\"name\":\"../etc/passwd\"}"` debe responder `400` y `GET /media/current` debe seguir mostrando la fuente previa sin cambios.

### Frame de alta resolución vs. frame de inferencia

Cada frame capturado (`frame_hires`) se mantiene sin modificar en todo momento (reservado para overlay/OCR futuro). Solo si `frame_hires` supera `BRIDGE_MAX_WIDTH` de ancho se deriva un `frame_infer` reducido (`cv2.resize` + `INTER_AREA`) exclusivamente para JPEG-encode e inferencia PaddleX; si el ancho está dentro del límite, `frame_infer` es el mismo frame (sin copia, sin costo extra). Las detecciones que devuelve PaddleX vienen en coordenadas de `frame_infer`, y el bridge las reescala de vuelta a coordenadas de `frame_hires` (bbox nativo) antes de enviarlas al adaptador — el contrato `epp_core.py` y el adaptador no ven ninguna diferencia.

### Medición de FPS/CPU

Dos formas de medir el impacto de estos ajustes:

- **Log en proceso**: cada `BRIDGE_METRICS_EVERY` frames inferidos, el bridge emite una línea `metrics infer_ms=... effective_fps=... dets=...` con la duración de inferencia del último frame de la ventana, el FPS efectivo promedio de la ventana y la cantidad de detecciones de ese frame.
- **`docker stats` (externo, sin cambios de código)**:

  ```bash
  docker stats vi-bridge
  ```

  Útil para comparar CPU%/memoria antes/después del cambio con una fuente `> 1280` px de ancho.

## Requisitos

- Docker Desktop (Windows/Mac) o Docker Engine + Compose v2
- FFmpeg en el host (solo para inyectar webcam)
- ~8 GB RAM recomendados; primera bajada de modelos PaddleX ~2–3 GB
- (Opcional) NVIDIA Container Toolkit para perfil `gpu`

## Arranque rápido

```bash
# 1) Clonar / entrar al repo
cd vision-intelligence
cp .env.example .env

# 2) Levantar stack
docker compose up --build
```

En otra terminal, inyectar webcam:

**Windows**

```bat
ffmpeg -list_devices true -f dshow -i dummy
inject_webcam.bat "USB Camera"
```

> Nota: en PaddleX 3.x el pipeline se llama `vehicle_attribute_recognition` (no existe `PP-Vehicle`; eso era PaddleDetection). Tipo/color + `track_id` salen siempre (IoU tracker en el bridge); la patente OCR es opcional vía el servicio `paddlex-ocr` — ver [sección OCR de patente](#ocr-de-patente-servicio-paddlex-ocr-opcional) más arriba.

**Linux**

```bash
chmod +x inject_webcam.sh
./inject_webcam.sh
```

Abrir:

| Recurso | URL |
|---------|-----|
| Dashboard AMIS | http://localhost:8000 |
| API eventos | http://localhost:8000/events |
| Health | http://localhost:8000/health |
| PaddleX (attr) | http://localhost:8080 |
| PaddleX (OCR, si `ENABLE_PLATE_OCR=true`) | http://localhost:8081 |
| WebRTC / HLS | ver [webrtc_config.md](webrtc_config.md) |

Detener:

```bash
docker compose down
```

## Modo demo (sin webcam ni GPU)

Valida consolidación de tracks + UI en segundos:

```bash
docker compose --profile demo up --build adapter bridge-demo
```

O con el stack completo y `DEMO_MODE=1` en `.env`.

## Perfiles Compose

| Comando | Efecto |
|---------|--------|
| `docker compose up --build` | MediaMTX + PaddleX CPU + Adapter + Bridge |
| `docker compose --profile demo up --build` | Bridge sintético |
| `docker compose --profile rules up --build` | + JetLinks `:8848` + rules-sink `:8850` |
| `docker compose --profile gpu up --build` | PaddleX con `runtime: nvidia` |

Con JetLinks real, setear en `.env`:

```env
JETLINKS_WEBHOOK_URL=http://jetlinks:8848/api/v1/vision/events
JETLINKS_API_KEY=demo
```

Todo `POST` hacia `JETLINKS_WEBHOOK_URL` incluye el header `x-api-key` con el valor de `JETLINKS_API_KEY` (default `demo`). Sin `JETLINKS_WEBHOOK_URL` configurada no se intenta ningún forward (modo MVP: solo log).

El contrato de `POST /webhook/rules` no cambia.

### rules-sink — capa de reglas headless (perfil `rules`)

Microservicio FastAPI adicional (`:8850`), alternativa reproducible/inspeccionable
a configurar reglas SQL a mano en la UI de `jetlinks`. Decide sobre los mismos
`PerceptionEvent` que el adaptador reenvía; no reemplaza ni modifica `jetlinks`
(ambos corren bajo el mismo perfil, en puertos distintos, sin conflicto).

| Endpoint | Auth | Descripción |
|----------|------|-------------|
| `POST /webhook/events` | `x-api-key` | Recibe un array de `PerceptionEvent`, evalúa la regla y devuelve `{received, alerted}` |
| `GET /health` | No | Healthcheck para Compose |
| `GET /alerts` | No | Últimas alertas en memoria (bounded, más nuevas primero) |

**Regla MVP**: alerta si algún `candidate_ids` empieza con `patente:` **o** si `confidence >= 0.7`.

Para activarlo:

```bash
docker compose --profile rules up --build
```

Y en `.env`, setear `JETLINKS_WEBHOOK_URL=http://rules-sink:8850/webhook/events` (mantener `JETLINKS_API_KEY == RULES_SINK_API_KEY`, ambos `demo` por default).

> **TODO auth (producción)**: la validación `x-api-key` es un secreto compartido estático pensado solo para red interna Docker (T3). No es autenticación de grado productivo — en producción reemplazar por OAuth/token real antes de exponer el servicio fuera de `epp-network`.

## Contrato epp-core (`epp_core.py`)

Garantías implementadas:

1. **Dos tiempos**: `occurred_at` (frame) + `observed_at` (proceso)
2. **Confianza comparable**: `confidence ∈ [0, 1]`
3. **Pistas, no veredictos**: `candidate_ids` (`patente:…`, `track:…`)
4. **Versionado**: `schema_version = "1.0-draft"`
5. **Consolidación**: `PerceptionEvent.consolidate_and_emit()` — votación ponderada por `score` para patente y color

El sweeper del adaptador emite al expirar el TTL del track (default 10 s) o si llega `finalized` / `track_lost`.

## Archivos clave

| Archivo | Rol |
|---------|-----|
| `epp_core.py` | Contrato Pydantic (portable) |
| `adapter.py` | FastAPI ingest / events / rules |
| `rtsp_bridge.py` | RTSP → PaddleX → ingest |
| `rules_sink.py` | Capa de reglas headless (perfil `rules`) |
| `amis_dashboard.json` | UI declarativa + ECharts |
| `dashboard.html` | Shell AMIS CDN |
| `docker-compose.yml` | Orquestación `epp-network` |
| `Dockerfile.*` | Imágenes adapter / bridge / paddlex |
| `inject_webcam.*` | Publicación RTSP desde el host |
| `webrtc_config.md` | WebRTC opcional |
| `videos_muestra/`, `imagenes_muestra/` | Muestras locales para el selector (gitignored, RO en compose) |

## Troubleshooting

**PaddleX tarda / descarga modelos**

```bash
docker compose logs -f paddlex
```

Es normal la primera vez (minutos + varios GB). El volume `vi-paddlex-models` cachea pesos.

**Webcam no se detecta (Windows)**

```bat
ffmpeg -list_devices true -f dshow -i dummy
```

Usar el nombre entre comillas exacto en `inject_webcam.bat`.

**El adaptador no recibe datos**

```bash
docker compose logs -f bridge
docker compose logs -f adapter
curl http://localhost:8000/health
```

Verificar que MediaMTX tiene publicación:

```bash
ffplay -rtsp_transport tcp rtsp://localhost:8554/webcam
```

**PaddleX caído**

El bridge no crashea: exponential backoff + señal `degraded` al adaptador. El dashboard muestra badge DEGRADADO.

**AMIS no carga gráficos**

Abrir la consola del navegador; confirmar `GET /events` y `GET /static/amis_dashboard.json`. La URL del browser es `localhost`, no `adapter`.

## Desarrollo local del adaptador (sin rebuild)

Los archivos Python/JSON del adaptador están montados con `--reload`. Editar `epp_core.py` / `adapter.py` / `amis_dashboard.json` y refrescar.

## Edge (RK3588) — roadmap

Mañana el mismo `epp_core.PerceptionEvent.consolidate_and_emit(detections)` corre en el SoC. Solo cambian variables de entorno (`RTSP_URL`, `PADDLEX_URL`). Cero rediseño del sobre común — habilita Fase 3 (Fusión con Document Intelligence).
