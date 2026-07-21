# detection/

## Para qué sirve

Código cliente de las capacidades de IA: vehículos, objetos COCO y patentes OCR.

## Cómo funciona

El `bridge/` orquesta estas carpetas sobre cada foto. Cada subcarpeta habla con
su servicio PaddleX y normaliza la respuesta a dicts que entiende el adapter.

```text
bridge → vehicles (:8080) + objects (:8082) → merge → plates (:8081 opcional)
```

## Entrada / salida

- **Entrada:** JPEG (bytes) o frame OpenCV según el módulo.
- **Salida:** listas de detecciones (`track_id`, `label`, `score`, `bbox`, …).

## Servicio / deps

Ver cada subcarpeta. Shared: `detection/common/` (tracker, geometry, preview).

## Archivos clave

| Carpeta | Capacidad |
|---------|-----------|
| [vehicles/](vehicles/) | Tipo/color de vehículo |
| [objects/](objects/) | COCO (incluye `person`) |
| [plates/](plates/) | OCR de patente |
| [common/](common/) | Utilidades compartidas |

## Qué no es

No levanta modelos ni Docker: solo clientes HTTP + normalización.
