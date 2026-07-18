#!/usr/bin/env bash
# Inyecta la webcam del host hacia MediaMTX (RTSP).
# Ejecutar en el HOST (fuera de Docker), con docker compose ya levantado.
set -euo pipefail

RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/webcam}"
DEVICE="${VIDEO_DEVICE:-/dev/video0}"

echo "==> Inyectando webcam ${DEVICE} → ${RTSP_URL}"
echo "    Ctrl+C para detener."

# Linux (V4L2). En macOS usar: -f avfoundation -i "0"
ffmpeg -hide_banner -loglevel info \
  -f v4l2 -input_format mjpeg -i "${DEVICE}" \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp \
  "${RTSP_URL}"
