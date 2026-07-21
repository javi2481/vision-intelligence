#!/bin/sh
# VI_PIPELINE/VI_PORT parametrizan la misma imagen para servir distintos
# pipelines PaddleX (attr por default, OCR via servicio paddlex-ocr).
# VI_USE_HPIP=1 añade --use_hpip (tras instalar hpi-cpu y validar ≥~1.5×).
set -eu
HPIP_ARGS=""
case "${VI_USE_HPIP:-0}" in
  1|true|TRUE|yes|YES) HPIP_ARGS="--use_hpip" ;;
esac
exec paddlex --serve \
  --pipeline "${VI_PIPELINE:-vehicle_attribute_recognition}" \
  --port "${VI_PORT:-8080}" \
  --device "${VI_DEVICE:-cpu}" \
  ${HPIP_ARGS}
