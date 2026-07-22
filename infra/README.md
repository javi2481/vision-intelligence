# infra/

## Para qué sirve

Imagen Docker compartida de PaddleX y su entrypoint. Los servicios
`paddlex*` usan la misma build y se diferencian por `VI_PIPELINE` / `VI_PORT`.

## Cómo funciona

1. `Dockerfile.paddlex` instala `paddlex[cv,serving,ocr]` **pineado a `==3.7.2`**
   (sin fallbacks `||`): las dos builds (`vision-intelligence/paddlex:local`
   y el resto de servicios `paddlex-*` que comparten esta imagen) deben usar
   siempre la misma versión exacta, para reproducibilidad.
2. Copia pipelines OCR v5 mobile + stubs scene lane/bdd_marks.
3. `entrypoint.paddlex.sh` ejecuta `paddlex --serve --pipeline … --port …`.
4. `VI_USE_HPIP=1` añade `--use_hpip` (High Performance Inference).

## Versión pineada de paddlex

- **Versión actual:** `3.7.2` (fijada en `infra/Dockerfile.paddlex`, sin `||`
  fallback a rangos abiertos).
- **Por qué pin exacto:** un fallback (`>=3.0.0` o similar) puede traer una
  versión nueva sin aviso en un rebuild y romper pipelines en producción
  silenciosamente. Ya hubo un incidente real: PP-OCRv6 (default en paddlex
  3.7.2) no es compatible con el motor de inferencia (PIR) de esta imagen
  base — por eso el pipeline OCR está pineado aparte a PP-OCRv5 mobile
  (`ocr_v5_mobile.yaml`). Sin versión exacta de `paddlex`, ese incidente se
  puede repetir en cualquier rebuild futuro sin que nadie lo note hasta que
  falla en runtime.

### Procedimiento de upgrade

1. Bump del pin en `infra/Dockerfile.paddlex` (`paddlex[cv,serving,ocr]==X.Y.Z`).
2. Rebuild: `docker compose build paddlex paddlex-ocr paddlex-objects`
   (o `docker compose build` para todo el perfil que uses).
3. Smoke:
   - **Notebook 8 GB:** solo `docker compose --profile default up` +
     verificar `/docs` u `/openapi.json` de cada servicio levantado
     (ver healthchecks en `docker-compose.yml`). El smoke extendido
     (`scripts/smoke_extended.sh`) **no corre en esta máquina** — RAM
     insuficiente para `extended`/`experimental`.
   - **Desktop 32 GB:** además correr `scripts/smoke_extended.sh` con
     `--profile extended` (y `experimental` si aplica) antes de aceptar
     el bump.
4. Confirmar especialmente que el pipeline OCR (`ocr_v5_mobile.yaml`) sigue
   sirviendo sin error de PIR/versión — es el punto de fricción conocido.
5. Recién ahí, commitear el bump del pin.

## Optimización (Fase 0)

### Benchmark

Con el stack up:

```bash
PYTHONPATH=. python3 scripts/benchmark_paddlex.py --image imagenes_muestra/TU_FOTO.jpg --rounds 5
```

Anotar `mean_s` por servicio. Criterio para HPIP: mejora ≥ ~1.5× vs baseline.

### HPIP

| Env | Efecto |
|-----|--------|
| `VI_USE_HPIP=0` (default) | Serve estándar |
| `VI_USE_HPIP=1` | `--use_hpip` en entrypoint |

Requiere plugin HPI en la imagen (`paddlex --install hpi-cpu` o imagen GPU con TensorRT).
Probar **por servicio** (no hace falta activarlo en todos).

### GPU

```bash
docker compose --profile gpu up --build
```

Usar `PADDLE_GPU_BASE_IMAGE` y, si aplica, backends `trt_fp16` vía config HPI.
El entrypoint exige `nvidia-smi` usable cuando `VI_DEVICE=gpu`; si no hay GPU/runtime, el contenedor falla al arrancar (en vez de quedar “up” sin servir).

### Límites de RAM (compose)

Ver README raíz (§ RAM del host). Anchors `x-limits-default` (8 GB / default) y
`x-limits-extended` (32 GB / extended+experimental).

### Modelos lite (CPU)

| Capacidad | Preferencia lite |
|-----------|------------------|
| OCR | Ya: PP-OCRv5 mobile (`OCR_PIPELINE_CONFIG`) |
| Objects | PicoDet vía config object_detection |
| Scene | PP-LiteSeg-T en YAML lane/bdd |
| Faces | BlazeFace (default face_detection) |

### Tune bridge

| Var | Default | Nota |
|-----|---------|------|
| `BRIDGE_MAX_WIDTH` | 960 | Bajar a 640 en CPU saturada |
| `HTTP_TIMEOUT` | 30 | Subir si scene/seg tarda |
| `OCR_TOPK` / `OCR_HTTP_TIMEOUT` | 3 / 5 | Limitar costo OCR |

## Entrada / salida

HTTP serving de cada pipeline según el servicio.

## Archivos clave

- `Dockerfile.paddlex`
- `entrypoint.paddlex.sh`
- [`scripts/benchmark_paddlex.py`](../scripts/benchmark_paddlex.py)
- [`scripts/smoke_extended.sh`](../scripts/smoke_extended.sh)

## Qué no es

No contiene lógica de producto Python. No es el bridge ni el adapter.
