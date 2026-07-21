#!/usr/bin/env bash
# Smoke del profile extended: health + flags tipicos.
# Uso: ./scripts/smoke_extended.sh
set -euo pipefail

ADAPTER="${ADAPTER_URL:-http://127.0.0.1:8000}"
echo "== adapter health =="
curl -fsS "$ADAPTER/health" | head -c 400
echo

for port_name in \
  "8080:vehicles" \
  "8081:ocr" \
  "8082:objects" \
  "8083:faces" \
  "8084:pedestrians" \
  "8085:scene" \
  "8086:pose" \
  "8087:face_id" \
  "8088:signs"
do
  port="${port_name%%:*}"
  name="${port_name##*:}"
  echo "== paddlex $name :$port =="
  if curl -fsS -m 3 "http://127.0.0.1:${port}/" >/dev/null 2>&1 \
    || curl -fsS -m 3 "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
    echo "  reachable"
  else
    echo "  not reachable (ok si el profile/servicio no está up)"
  fi
done

echo "== events sample =="
curl -fsS "$ADAPTER/events?limit=5" | head -c 800
echo
echo "Smoke done. Activá ENABLE_* en .env y subí una foto al dashboard."
