# ADR: consolidar pipelines PaddleX en menos procesos

- **Estado:** propuesta (spike, sin implementación en Sprint 2)
- **Fecha:** 2026-07-21
- **Contexto:** perfil `extended` levanta 10+ contenedores PaddleX; cada uno carga runtime + pesos. En host 8 GB solo cabe `default`; en 32 GB extended es viable pero costoso en RAM.

## Pregunta

¿Se pueden agrupar modelos livianos en menos procesos `paddlex --serve` para reducir huella, sin romper el diseño pedagógico **1 carpeta = 1 capacidad = 1 servicio**?

## Hechos

- El serving de PaddleX 3.x es históricamente **un pipeline por proceso** (`VI_PIPELINE` + puerto).
- Compose ya parametriza la misma imagen (`infra/Dockerfile.paddlex`) por env; el costo es N procesos, no N imágenes distintas.
- El valor de 1:1 carpeta/servicio: aislamiento de fallos, perfiles Compose, onboarding por capacidad, flags `ENABLE_*` independientes.

## Opciones

1. **Mantener 1:1** (recomendado corto plazo): techos `mem_limit` + profiles; no fusionar.
2. **Agrupar experimental** (signos livianos / scene_cls / small) en 1–2 procesos multi-pipeline *si* PaddleX lo permite de forma estable — requiere spike de serving.
3. **Sidecar único con router HTTP interno** — más ingeniería, choca con “orquestar no inventar”.
4. **(a) FastAPI custom multi-pipeline vía `create_pipeline`** — en vez de N procesos
   `paddlex --serve`, un único proceso FastAPI que instancia varios pipelines con la
   API Python de PaddleX (`paddlex.create_pipeline(...)`) y expone rutas propias por
   capacidad. Reduce huella de runtime duplicado (1 proceso Python en vez de N), pero
   pierde el aislamiento de fallos por servicio (un crash de un pipeline puede tumbar
   el proceso completo) y requiere mantener el router/serialización a mano en vez de
   delegar en `paddlex --serve`. Mayor superficie propia de código — tensiona con
   “orquestar no inventar” más que la opción 2, pero menos que un sidecar completo.
5. **(b) Triton Inference Server** — exportar los modelos PaddleX a un backend servible
   por Triton (ONNX/Paddle backend) y correr un solo servidor Triton multi-modelo.
   Beneficio: gestión de memoria/batching madura y un solo proceso para muchos modelos.
   Costo: introduce una pieza de infraestructura nueva y pesada (no es "orquestar",
   es agregar una dependencia mayor), exportación de modelos no trivial para todos los
   pipelines PaddleX usados acá, y curva de aprendizaje/operación adicional para un
   proyecto pedagógico. Candidato solo si el spike de la opción 2/4 no alcanza y el
   costo de RAM sigue siendo bloqueante en un host real de producción.

### RAM medida (placeholder — pendiente de spike)

*(Sin medición real todavía. Este ADR es una propuesta/spike sin implementación en
Sprint 2 — ver Estado. Completar esta tabla cuando se corra el spike con
`scripts/benchmark_paddlex.py` + `docker stats` en el host real.)*

| Escenario | RSS por proceso (medido) | RSS total perfil | Notas |
|---|---|---|---|
| `default` (paddlex + objects + ocr) | TBD | TBD | Notebook 8 GB — único perfil viable hoy |
| `extended` (+ faces/pedestrians/scene/pose/face_id/signs) | TBD | TBD | Desktop 32 GB |
| `experimental` (+ scene_cls/instances/small/anomaly/open_vocab) | TBD | TBD | Desktop 32 GB, opt-in |
| Opción 4 (FastAPI multi-pipeline, N modelos en 1 proceso) | TBD | TBD | Requiere spike de serving separado |
| Opción 5 (Triton) | TBD | TBD | Requiere exportación de modelos, spike separado |

### Nota RAM — notebook 8 GB (restricción vigente)

El perfil `default` (`x-limits-default`: `mem_limit: 1200m` × 3 servicios paddlex)
debe convivir en la máquina de 8 GB con `adapter` y `bridge` (sin `mem_limit`
explícito hoy, pero corriendo en el mismo host) — y, a partir del addendum SPA
(Fase 1), también con el contenedor/servido de la SPA en `/app/`. Cualquier ajuste
de `mem_limit` en `x-limits-default` o adición de nuevos servicios al perfil
`default` debe seguir cabiendo en ese presupuesto de 8 GB junto con SO + Docker
Desktop/WSL2 + navegador. El perfil `extended` (`x-limits-extended`, 32 GB) **no
corre en la máquina de desarrollo diaria** — su smoke (`scripts/smoke_extended.sh`)
está fuera del ciclo de verificación habitual en 8 GB (ver `infra/README.md`).

## Decisión (Sprint 2)

**No merge de servicios.** Entregable de este ADR: documentar el trade-off y dejar el spike como follow-up solo si, tras medir con `scripts/benchmark_paddlex.py` + `docker stats`, la RAM de extended sigue siendo bloqueante en el desktop 32 GB *después* de los `mem_limit`.

## Criterio para reabrir

- Un host target real < 16 GB necesita más de 3 capacidades a la vez, **o**
- PaddleX documenta/soporta multi-pipeline serve estable en un proceso.

## Consecuencias

- Sigue el mapa carpeta ↔ capacidad ↔ servicio del README.
- La reducción de huella inmediata es operativa (`mem_limit`, no levantar extended en 8 GB), no arquitectónica.
