# Vision Intelligence — Producto B (Sprint 1)

Orquestar, no inventar. Pipeline Docker-first **foto-only**:
**Foto → detection (vehicles + objects COCO + plates OCR) → adapter/epp_core → AMIS**.

## Mapa carpeta ↔ capacidad ↔ servicio

| Carpeta | Capacidad | Servicio Compose |
|---------|-----------|------------------|
| [detection/vehicles/](detection/vehicles/) | Tipo/color de vehículo | `paddlex` `:8080` |
| [detection/objects/](detection/objects/) | COCO (incluye **person**) | `paddlex-objects` `:8082` |
| [detection/plates/](detection/plates/) | OCR patente (opcional) | `paddlex-ocr` `:8081` |
| [detection/common/](detection/common/) | Tracker, geometry, preview | — |
| [bridge/](bridge/) | Orquestador foto → ingest/preview | `bridge` |
| [adapter/](adapter/) | Media, consolidación, API | `adapter` `:8000` |
| [adapter/ui/](adapter/ui/) | Panel AMIS | (estáticos del adapter) |
| [rules/](rules/) | Alertas headless | `rules-sink` (profile `rules`) |
| [infra/](infra/) | Imagen PaddleX compartida | build de `paddlex*` |
| [tests/](tests/) | Unit tests | — |

Cada carpeta tiene su propio `README.md` (para qué / cómo / I-O / deps).

> **Personas:** no hay pipeline aparte. `person` es clase COCO en
> [detection/objects/](detection/objects/).

## Arquitectura

```text
[Upload / imagenes_muestra] --> [adapter] <--poll-- [bridge]
                                                      |
                        +-----------------------------+------------------+
                        v                             v                  v
              detection/vehicles            detection/objects    detection/plates
                 paddlex:8080              paddlex-objects:8082   paddlex-ocr:8081
                        |                             |                  |
                        +------------ merge + OCR opcional --------------+
                                          |
                                   POST /ingest + /preview/frame
                                          v
                              PerceptionEvent → AMIS (/events)
```

## Arranque rápido

```bash
cp .env.example .env
docker compose up --build
```

| Recurso | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| Events | http://localhost:8000/events |
| Health | http://localhost:8000/health |
| PaddleX vehicles | http://localhost:8080 |
| PaddleX OCR | http://localhost:8081 |
| PaddleX objects | http://localhost:8082 |

## Flujo foto

1. Subí JPG desde el panel o copiá a `imagenes_muestra/`.
2. Adapter auto-selecciona; bridge polea `/media/current`.
3. Inferencia vehicles ∥ objects → merge → plates si `ENABLE_PLATE_OCR=true`.
4. Overlay EN + eventos en el panel. **Limpiar foto** → bridge idle.

## Perfiles Compose

| Comando | Efecto |
|---------|--------|
| `docker compose up --build` | paddlex* + adapter + bridge |
| `docker compose --profile demo up --build` | bridge sintético |
| `docker compose --profile rules up --build` | + JetLinks + rules-sink |
| `docker compose --profile gpu up --build` | PaddleX GPU |

## Variables útiles

Ver [`.env.example`](.env.example). Destacadas: `ENABLE_PLATE_OCR`, `MEDIA_DIR`,
`PADDLEX_*`, `BRIDGE_MAX_WIDTH`, `VI_ENV`.

## Tests

```bash
PYTHONPATH=. python3 tests/test_bridge_helpers.py
PYTHONPATH=. python3 tests/test_epp_core.py
PYTHONPATH=. python3 tests/test_adapter_media.py
```

Detalle en [tests/README.md](tests/README.md).

## Contrato epp-core

Portable en [adapter/epp_core.py](adapter/epp_core.py): entra dict de detección,
sale `PerceptionEvent` (votación patente/color/`class_name`). Sin reglas de negocio.

## Troubleshooting

```bash
docker compose logs -f bridge
docker compose logs -f adapter
curl http://localhost:8000/media/current
```

Sin foto activa el bridge queda idle (esperado).
