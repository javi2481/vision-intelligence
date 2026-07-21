# detection/pedestrians/

## Para qué sirve

Atributos de personas (género, edad, ropa, …) vía PaddleX
`pedestrian_attribute_recognition`. Enriquece dets COCO `person` de `objects/`.

## Cómo funciona

1. Opcional: `ENABLE_PEDESTRIAN_ATTRS=true`.
2. Inferencia en paralelo al resto.
3. `merge_person_attributes` empareja por IoU con labels `person`.
4. Sin match: emite det `person` extra con attrs.

## Entrada / salida

- **Entrada:** `jpeg: bytes` + lista de dets objects.
- **Salida:** dets objects enriquecidas con clave `person: {...}`.
- Fallo HTTP → `None` (no degrada).

## Servicio / deps

| Item | Valor |
|------|--------|
| Compose | `paddlex-pedestrians` (profile `extended`) |
| Puerto | `8084` |
| Env | `PADDLEX_PEDESTRIANS_URL`, `ENABLE_PEDESTRIAN_ATTRS` |

## Archivos clave

- `client.py` — `infer_pedestrian_attrs`, `merge_person_attributes`.

## Qué no es

No es un detector de personas. El bbox `person` sigue en `objects/`.
No crear carpeta `persons/`.
