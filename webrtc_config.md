# Visualización WebRTC en vivo (opcional) — MediaMTX

MediaMTX ya expone **WebRTC** además de RTSP. No hace falta código propio:
el navegador consume el mismo path `webcam` que inyecta FFmpeg.

## Requisitos

1. `docker compose up` con el servicio `mediamtx` corriendo.
2. Webcam inyectada (`inject_webcam.bat` / `inject_webcam.sh`).
3. Puertos publicados (ver `docker-compose.yml`):
   - `8554` — RTSP
   - `8889` — WebRTC (WHIP/WHEP)
   - `8888` — HLS (alternativa)

## Abrir en el navegador

Con la imagen oficial `bluenviron/mediamtx`:

| Protocolo | URL típica |
|-----------|------------|
| WebRTC (WHEP playback) | `http://localhost:8889/webcam` |
| HLS | `http://localhost:8888/webcam` |
| RTSP (VLC / ffplay) | `rtsp://localhost:8554/webcam` |

Si `8889/webcam` no abre un player embebido en tu versión de MediaMTX,
usa un cliente WHEP (p. ej. la página de demos de MediaMTX) apuntando a:

```text
http://localhost:8889/webcam/whep
```

## Verificar que el path existe

```bash
docker compose logs mediamtx
ffplay -rtsp_transport tcp rtsp://localhost:8554/webcam
```

## Notas de producción

- WebRTC es **solo visualización**; la IA sigue leyendo RTSP (bridge → PaddleX).
- No mezclar WebRTC en el adaptador FastAPI: separación de capas intacta.
- En LAN corporativa, publicar solo `8889` hacia clientes autorizados.
- Para edge (RK3588), MediaMTX + WHEP sigue siendo válido sin reescribir el adaptador.

## Activación

No hay flag extra: con `mediamtx` healthy e inyección de webcam, WebRTC queda disponible.
Desactivar = no inyectar webcam o no publicar el puerto `8889`.
