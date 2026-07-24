# imagenes_muestra/ — media + eval fixtures

Local image mount used by adapter/bridge (`./imagenes_muestra:/media/images`).
Heavy JPGs are gitignored; this README is the only tracked file.

## Layout (after eval download)

```text
imagenes_muestra/
  fo_objects_0001.jpg      # flat root — adapter media scan sees these
  fo_vehicles_0001.jpg
  fo_ocr_plates_0001.jpg
  packs/<suite>/*.jpg      # nested copies (not required for SPA scan)
  gt/<suite>.json          # ground truth per suite
  gt/manifest.json         # sha256 + seed=51
  failures/<suite>/*.json  # Tier A/B miss artifacts from eval
```

## Accuracy harness (local gate — not CI)

Host-only deps:

```bash
python -m pip install -r scripts/requirements-eval.txt
```

Download fixtures (seed=51, ~15–20 samples/suite):

```bash
# Core local gate (default)
PYTHONPATH=. python scripts/download_paddlex_eval.py --packs core --out imagenes_muestra

# Core + Extended + Experimental
PYTHONPATH=. python scripts/download_paddlex_eval.py --packs all --out imagenes_muestra
```

Evaluate live localhost PaddleX (stack already up):

```bash
PYTHONPATH=. python scripts/eval_paddlex_fixtures.py --packs core --out imagenes_muestra
PYTHONPATH=. python scripts/eval_paddlex_fixtures.py --packs all --out imagenes_muestra
# Subset: --only / --pipelines signs,faces,anomaly
```

| Layer | Suites | Notes |
|-------|--------|-------|
| Core | `objects` :8082, `vehicles` :8080, `ocr_plates` :8081 | Local accuracy gate |
| Extended | `signs` :8088, `faces` :8083, `pose` :8086, `pedestrians` :8084 (Tier B), `scene` :8085 (Tier B), `ocr_text` :8081 | Opt-in |
| Experimental | `face_id` :8087, `scene_cls` :8089, `instances` :8090, `small_objects` :8091, `anomaly` :8092 (Tier C smoke), `open_vocab` :8093 | Opt-in; Core never requires these |

- `ocr_plates` / `ocr_text` / `faces` / `anomaly` / `face_id` / `scene_cls` are **synthetic** (Pillow); FO COCO for detection suites
- `face_id` also writes `gt/face_id_gallery.json` (enrollment manifest)
- Honest Tier B where no pixel/class GT (`scene`, `open_vocab`, `scene_cls`, `face_id`)
- Baseline `scripts/eval_baseline.json` is **manual** — never auto-overwritten
- Exit non-zero on threshold breach or baseline regression (Core bars always; Extended/Exp only when those suites run)

### Manual E2E checklist

1. `docker compose up --build` (default profile: vehicles/objects/ocr; enable profiles for Extended/Exp ports)
2. Pause media-watch or accept thrash while downloading
3. `download_paddlex_eval.py --packs core` (or `--packs all`)
4. `eval_paddlex_fixtures.py --packs core` (or `--packs all` / `--pipelines …`)
5. Inspect `scripts/eval_report.json` + `failures/<suite>/`
6. If intentional model upgrade improved metrics, copy metrics into `eval_baseline.json` by hand

## Media-watch thrash

Bulk flat writes can overload the adapter watch. If the UI thrashs:

1. Pause / stop bridge+adapter briefly, or remove eval JPGs from the root
2. Prefer keeping pack caps (~20/suite; default Core)
3. Clear `fo_*` root JPGs between experiments; keep `packs/`/`gt/` if re-flattening

## Licenses

- COCO samples: follow [COCO dataset](https://cocodataset.org/#termsofuse) terms when downloading via FiftyOne
- Synthetic OCR plates: generated locally; no third-party plate photos

## What this is not

Not a CI gate. Not a Compose mount change. FiftyOne is never installed into Docker images.
