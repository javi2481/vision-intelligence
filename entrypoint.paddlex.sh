#!/bin/sh
# VI_PIPELINE/VI_PORT parametrizan la misma imagen para servir distintos
# pipelines PaddleX (attr por default, OCR via servicio paddlex-ocr).
exec paddlex --serve \
  --pipeline "${VI_PIPELINE:-vehicle_attribute_recognition}" \
  --port "${VI_PORT:-8080}" \
  --device "${VI_DEVICE:-cpu}"
