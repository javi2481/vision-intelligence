# Vision Intelligence — Producto B (Sprint 1)

Orquestar, no inventar. Pipeline Docker-first **foto-only**:
**Foto → detection/* → adapter/epp_core → AMIS**.

## Mapa carpeta ↔ capacidad ↔ servicio

| Carpeta | Capacidad | Servicio Compose |
|---------|-----------|------------------|
| [detection/vehicles/](detection/vehicles/) | Tipo/color de vehículo | `paddlex` `:8080` |
| [detection/objects/](detection/objects/) | COCO (incluye **person**) | `paddlex-objects` `:8082` |
| [detection/plates/](detection/plates/) | OCR patente (opcional) | `paddlex-ocr` `:8081` |
| [detection/faces/](detection/faces/) | Rostros (opcional) | `paddlex-faces` `:8083` (profile `extended`) |
| [detection/pedestrians/](detection/pedestrians/) | Attrs persona (opcional) | `paddlex-pedestrians` `:8084` (profile `extended`) |
| [detection/scene/](detection/scene/) | Escena vial (opcional) | `paddlex-scene` `:8085` (profile `extended`) |
| [detection/pose/](detection/pose/) | Keypoints (opcional) | `paddlex-pose` `:8086` (`extended`) |
| [detection/text/](detection/text/) | OCR carteles (opcional) | reusa `paddlex-ocr` `:8081` |
| [detection/face_id/](detection/face_id/) | Identidad facial (opcional) | `paddlex-face-id` `:8087` (`extended`) |
| [detection/signs/](detection/signs/) | Señales (opcional) | `paddlex-signs` `:8088` (`extended`) |
| [detection/scene_cls/](detection/scene_cls/) … [open_vocab/](detection/open_vocab/) | Medio fit (GATE) | `:8089`–`:8093` (`experimental`) |
| [detection/common/](detection/common/) | Tracker, geometry, preview | — |
| [bridge/](bridge/) | Orquestador foto → ingest/preview | `bridge` |
| [adapter/](adapter/) | Media, consolidación, API | `adapter` `:8000` |
| [adapter/ui/](adapter/ui/) | Panel AMIS | (estáticos del adapter) |
| [rules/](rules/) | Alertas headless | `rules-sink` (profile `rules`) |
| [infra/](infra/) | Imagen PaddleX compartida | build de `paddlex*` |
| [tests/](tests/) | Unit tests | — |

Cada carpeta tiene su propio `README.md` (para qué / cómo / I-O / deps).

> **Personas:** el bbox `person` es clase COCO en [detection/objects/](detection/objects/).
> Los atributos van en [detection/pedestrians/](detection/pedestrians/) (no hay `persons/`).

## Arquitectura

```text
[Upload / imagenes_muestra] --> [adapter] <--poll-- [bridge]
                                                      |
              +-----------+-----------+---------------+---+--------+
              v           v           v               v   v        v
         vehicles     objects      plates?        faces ped.    scene
         :8080        :8082        :8081          :8083 :8084   :8085
              |           |           |               |   |        |
              +-----------+---- merge + OCR + extended +---+--------+
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

### RAM del host

| Máquina | Qué levantar | `mem_limit` paddlex |
|---------|--------------|---------------------|
| Notebook **~8 GB** | Solo profile **default** (~3 servicios) | ~1.2g c/u (anchor `limits-default`) |
| Desktop **~32 GB** | **default + extended** | ~2g c/u (`limits-extended`) |
| Experimental | Solo con ≥32 GB y opt-in | mismos techos extended |

No levantes `--profile extended` en 8 GB: aunque haya techos, no cabe el presupuesto de modelos.
Si un servicio muere `OOMKilled`, subí su `mem_limit` en compose o bajá `BRIDGE_MAX_WIDTH`.

Capacidades extended (rostros / attrs persona / escena) — **desktop 32 GB**:

```bash
# En .env: ENABLE_FACE_DETECTION=true ENABLE_PEDESTRIAN_ATTRS=true ENABLE_SCENE_SEG=true
docker compose --profile extended up --build
```

| Recurso | URL |
|---------|-----|
| Dashboard | http://localhost:8000 |
| Events | http://localhost:8000/events |
| Health | http://localhost:8000/health |
| PaddleX vehicles | http://localhost:8080 |
| PaddleX OCR | http://localhost:8081 |
| PaddleX objects | http://localhost:8082 |
| PaddleX faces | http://localhost:8083 (extended) |
| PaddleX pedestrians | http://localhost:8084 (extended) |
| PaddleX scene | http://localhost:8085 (extended) |
| PaddleX pose | http://localhost:8086 (extended) |
| PaddleX face_id | http://localhost:8087 (extended) |
| PaddleX signs | http://localhost:8088 (extended) |
| Medio fit | http://localhost:8089–8093 (`experimental`) |

## Flujo foto

1. Subí JPG desde el panel o copiá a `imagenes_muestra/`.
2. Adapter auto-selecciona; bridge polea `/media/current`.
3. Inferencia vehicles ∥ objects (+ faces/pedestrians/scene si flags) → merge → plates si OCR on.
4. Overlay EN + eventos en el panel. **Limpiar foto** → bridge idle.

## Perfiles Compose

| Comando | Efecto |
|---------|--------|
| `docker compose up --build` | paddlex* default + adapter + bridge |
| `docker compose --profile extended up --build` | + faces, pedestrians, scene, pose, face_id, signs |
| `docker compose --profile experimental up --build` | + scene_cls, instances, small_objects, anomaly, open_vocab (GATE) |
| `docker compose --profile demo up --build` | bridge sintético |
| `docker compose --profile rules up --build` | + JetLinks + rules-sink |
| `docker compose --profile gpu up --build` | PaddleX GPU |

## Variables útiles

Ver [`.env.example`](.env.example). Destacadas: `ENABLE_PLATE_OCR`,
`ENABLE_FACE_DETECTION`, `ENABLE_PEDESTRIAN_ATTRS`, `ENABLE_SCENE_SEG`,
`ENABLE_POSE`, `ENABLE_SCENE_OCR`, `ENABLE_FACE_ID`, `ENABLE_SIGNS`,
flags `experimental` (`ENABLE_SCENE_CLS` …), `VI_USE_HPIP`,
`MEDIA_DIR`, `PADDLEX_*`, `BRIDGE_MAX_WIDTH`, `VI_ENV`.

## Tests

```bash
PYTHONPATH=. python3 tests/test_bridge_helpers.py
PYTHONPATH=. python3 tests/test_epp_core.py
PYTHONPATH=. python3 tests/test_adapter_media.py
```

Detalle en [tests/README.md](tests/README.md).

## Contrato epp-core

Portable en [adapter/epp_core.py](adapter/epp_core.py): entra dict de detección,
sale `PerceptionEvent` (votación patente/color/`class_name`/scene). Sin reglas de negocio.

## Troubleshooting

```bash
docker compose logs -f bridge
docker compose logs -f adapter
curl http://localhost:8000/media/current
```

Sin foto activa el bridge queda idle (esperado).
