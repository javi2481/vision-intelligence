# tests/

## Para qué sirve

Tests stdlib (`unittest`) de helpers de detection, bridge.media, epp_core y
media del adapter. No son CI obligatoria localmente; GitHub Actions corre los
mismos tres archivos en `.github/workflows/ci.yml`.

## Cómo funciona

Desde la raíz del repo, con deps instaladas (`opencv`, `numpy`, `pydantic`,
`fastapi`, …):

```bash
PYTHONPATH=. python3 tests/test_bridge_helpers.py
PYTHONPATH=. python3 tests/test_epp_core.py
PYTHONPATH=. python3 tests/test_adapter_media.py
```

Con vendor local (si existe `.vendor/`):

```bash
PYTHONPATH=".vendor:." python3 tests/test_adapter_media.py
```

## Fixtures

| Archivo | Descripción |
|---------|-------------|
| `fixtures/sample.jpg` | JPEG mínimo válido (1×1) para pruebas de media/preview sin depender de PIL |

Generado como bytes JFIF embebidos (no requiere Pillow). Se puede regenerar con
cualquier encoder que escriba un `.jpg` pequeño en esa ruta.

## Entrada / salida

Sin servicios Docker: solo funciones puras / helpers.

## Servicio / deps

Ninguno. Requiere packages de `bridge/requirements.txt` + `adapter/requirements.txt`.

## Archivos clave

| Test | Cubre |
|------|--------|
| `test_bridge_helpers.py` | geometry, preview, vehicles, objects, media |
| `test_epp_core.py` | consolidación / entity_type |
| `test_adapter_media.py` | auto-select mtime |

## Qué no es

No sustituye smoke E2E con `docker compose up`.
