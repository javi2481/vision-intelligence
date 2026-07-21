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

## Decisión (Sprint 2)

**No merge de servicios.** Entregable de este ADR: documentar el trade-off y dejar el spike como follow-up solo si, tras medir con `scripts/benchmark_paddlex.py` + `docker stats`, la RAM de extended sigue siendo bloqueante en el desktop 32 GB *después* de los `mem_limit`.

## Criterio para reabrir

- Un host target real < 16 GB necesita más de 3 capacidades a la vez, **o**
- PaddleX documenta/soporta multi-pipeline serve estable en un proceso.

## Consecuencias

- Sigue el mapa carpeta ↔ capacidad ↔ servicio del README.
- La reducción de huella inmediata es operativa (`mem_limit`, no levantar extended en 8 GB), no arquitectónica.
