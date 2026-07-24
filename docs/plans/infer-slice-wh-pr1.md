# PR1 measurement: `INFER_SLICE_WH` proposal

**Date:** 2026-07-24  
**Baseline:** `BRIDGE_MAX_WIDTH` default **960** (unchanged in PR1).  
**Command:** `PYTHONPATH=. python scripts/measure_infer_slice.py --out imagenes_muestra --live`

## Findings

| Signal | Result |
|--------|--------|
| Core fixture source widths | Mostly ≤640 (pack images already small) |
| Bridge infer width after `maybe_resize_for_infer` | **max 640** on this pack (no downscale to 960 exercised) |
| Live `result.image` size (vehicles/objects) | Matches bridge JPEG `wh` when present |
| GT bbox width (hires) | See regenerable `scripts/infer_slice_measure.json` (not committed) |

## Proposal

**`INFER_SLICE_WH = 640`** for PR2 default.

Rationale: measured effective JPEG width into PaddleX on the Core pack is ≤640; round tile size at that scale. PR2 still raises `BRIDGE_MAX_WIDTH` to 1920 separately so large photos become hires for tiling; tiles stay at the measured slice size unless retuned.

## Non-goals (PR1)

- Did **not** change `BRIDGE_MAX_WIDTH` default (still 960).
- Did **not** invent an internal PaddleX tensor size beyond JPEG/`result.image` evidence.
