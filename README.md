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
| `BRIDGE_FPS` | `1` | Frecuencia de muestreo para JPEG-encode + inferencia PaddleX. |
| `BRIDGE_MAX_WIDTH` | `1280` | Ancho máximo de la imagen que se envía a inferencia. Sobre este umbral se reduce solo la copia de inferencia (ver abajo); a la par o por debajo no hay resize. |
| `BRIDGE_METRICS_EVERY` | `30` | Cada cuántos frames inferidos se emite una línea de métricas en el log. |

### Frame de alta resolución vs. frame de inferencia

Cada frame capturado (`frame_hires`) se mantiene sin modificar en todo momento (reservado para overlay/OCR futuro). Solo si `frame_hires` supera `BRIDGE_MAX_WIDTH` de ancho se deriva un `frame_infer` reducido (`cv2.resize` + `INTER_AREA`) exclusivamente para JPEG-encode e inferencia PaddleX; si el ancho está dentro del límite, `frame_infer` es el mismo frame (sin copia, sin costo extra). Las detecciones que devuelve PaddleX vienen en coordenadas de `frame_infer`, y el bridge las reescala de vuelta a coordenadas de `frame_hires` (bbox nativo) antes de enviarlas al adaptador — el contrato `epp_core.py` y el adaptador no ven ninguna diferencia.

### Medición de FPS/CPU

Dos formas de medir el impacto de estos ajustes:

- **Log en proceso**: cada `BRIDGE_METRICS_EVERY` frames inferidos, el bridge emite una línea `metrics infer_ms=... encode_ms=... effective_fps=... resized=... infer_w=...` con la duración de inferencia/encode del último frame de la ventana y el FPS efectivo promedio de la ventana.
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

> Nota: en PaddleX 3.x el pipeline se llama `vehicle_attribute_recognition` (no existe `PP-Vehicle`; eso era PaddleDetection). La patente OCR queda para una fase siguiente; hoy salen tipo/color + `track_id` vía IoU tracker en el bridge.

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
| PaddleX | http://localhost:8080 |
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
