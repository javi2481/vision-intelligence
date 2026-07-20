#!/usr/bin/env bash
# Publica un video de muestra en loop hacia MediaMTX (camino "vivo" reproducible).
# Ejecutar en el HOST con docker compose ya levantado (mediamtx up).
#
# Uso:
#   ./publish_sample.sh
#   ./publish_sample.sh videos_muestra/Brasil6.mp4
#
# Luego en el panel: "Limpiar selección (volver a RTSP vivo)".
set -euo pipefail

RTSP_URL="${RTSP_URL:-rtsp://localhost:8554/webcam}"
SAMPLE="${1:-videos_muestra/Brasil6.mp4}"

if [[ ! -f "${SAMPLE}" ]]; then
  echo "ERROR: no existe ${SAMPLE}" >&2
  exit 1
fi

echo "==> Publicando ${SAMPLE} en loop → ${RTSP_URL}"
echo "    En el panel: limpiar selección de muestra local."
echo "    Ctrl+C para detener."

ffmpeg -hide_banner -loglevel info \
  -re -stream_loop -1 -i "${SAMPLE}" \
  -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -an \
  -f rtsp -rtsp_transport tcp \
  "${RTSP_URL}"
