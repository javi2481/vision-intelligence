# PR2 smoke: vehicles Core — tiled vs baseline paths

**Date:** 2026-07-24  
**Host:** `PADDLEX_URL=http://127.0.0.1:8080`, pack Core `--only vehicles`, 19 fixtures.  
**Code:** `feat/tiling-pr2-slicer` (`infer_tiled_sync` / eval harness).

## Commands

```text
PYTHONPATH=. python scripts/eval_paddlex_fixtures.py --packs core --only vehicles --report scripts/eval_report_pr2_direct.json
PYTHONPATH=. python scripts/eval_paddlex_fixtures.py --packs core --only vehicles --via-bridge-preprocess --report scripts/eval_report_pr2_bridge960.json
PYTHONPATH=. python scripts/eval_paddlex_fixtures.py --packs core --only vehicles --via-tiled-sync --report scripts/eval_report_pr2_tiled.json
```

## Results (bbox_match_rate / schema_ok_rate)

| Path | bbox_match_rate | schema_ok_rate | tp / gt |
|------|-----------------|----------------|---------|
| Direct JPEG bytes (no re-encode) | **0.4691** | 1.0 | 38 / 81 |
| `--via-bridge-preprocess` (imdecode + `encode_jpeg`) | **0.358** | 1.0 | 29 / 81 |
| `--via-tiled-sync` (slicer → `encode_jpeg` per tile) | **0.358** | 1.0 | 29 / 81 |

Reports (local, not necessarily committed): `scripts/eval_report_pr2_*.json`.

## Interpretation (no invented internal sizes)

- On this Core pack (fixtures mostly ≤640; see PR1 `infer-slice-wh-pr1.md`), tiled match rate **equals** the bridge-preprocess path.
- Both are **below** direct file bytes; the shared difference is **JPEG re-encode** (`encode_jpeg` / `JPEG_QUALITY`), not multi-tile NMS (most images are a single tile).
- PaddleX serving docs do **not** document an internal tensor resize; we do not claim one here.
- `ENABLE_INFER_TILING` remains **false** by default until measured on larger (hires / 1920) photos where multi-tile is the point.

## Stack defaults applied in PR2

- `INFER_OVERLAP_WH=100` — [supervision 0.28 InferenceSlicer](https://supervision.roboflow.com/0.28.0/detection/tools/inference_slicer/) default.
- `INFER_SLICE_WH=640` — PR1 measurement proposal.
- NMS-A: `NON_MAX_SUPPRESSION` + `IOU` + `thread_workers=1` (same API defaults).
