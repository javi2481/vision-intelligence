#!/usr/bin/env python3
"""PR1 measurement: bridge resize + GT bbox widths → propose INFER_SLICE_WH.

Runs offline for histograms (no PaddleX required). Optional --live posts a
sample JPEG to vehicles/objects and records decoded result.image size if present.

Baseline assumes BRIDGE_MAX_WIDTH default 960 (do not raise in PR1).

Usage:
  PYTHONPATH=. python scripts/measure_infer_slice.py --out imagenes_muestra
  PYTHONPATH=. python scripts/measure_infer_slice.py --out imagenes_muestra --live
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detection.common.geometry import (  # noqa: E402
    BRIDGE_MAX_WIDTH,
    encode_jpeg,
    maybe_resize_for_infer,
)


def _percentile(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _bbox_width(bbox: list[float]) -> float:
    return abs(float(bbox[2]) - float(bbox[0]))


def _load_core_gt(out: Path, suites: list[str]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for suite in suites:
        path = out / "gt" / f"{suite}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing GT: {path}")
        data[suite] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _histogram(widths: list[float], bin_edges: list[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for w in widths:
        placed = False
        for i in range(len(bin_edges) - 1):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if lo <= w < hi:
                counts[f"[{lo},{hi})"] += 1
                placed = True
                break
        if not placed:
            counts[f">={bin_edges[-1]}"] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def _decode_result_image_size(data: dict[str, Any]) -> Optional[tuple[int, int]]:
    result = data.get("result") if isinstance(data, dict) else None
    if not isinstance(result, dict):
        return None
    img = result.get("image")
    if not isinstance(img, str) or not img:
        return None
    try:
        raw = base64.b64decode(img)
    except Exception:  # noqa: BLE001
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    h, w = frame.shape[:2]
    return int(w), int(h)


def _post_image(url: str, jpeg: bytes, timeout: float) -> dict[str, Any]:
    payload = json.dumps({"image": base64.b64encode(jpeg).decode("ascii")}).encode()
    req = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def measure(
    out: Path,
    *,
    live: bool,
    timeout: float,
) -> dict[str, Any]:
    suites = ["vehicles", "objects"]
    gt_by_suite = _load_core_gt(out, suites)

    bridge_widths: list[int] = []
    source_widths: list[int] = []
    gt_widths_hires: list[float] = []
    gt_widths_infer: list[float] = []
    live_result_sizes: list[dict[str, Any]] = []

    seen_files: set[str] = set()
    for suite, gt in gt_by_suite.items():
        for fx in gt.get("fixtures") or []:
            rel = fx["file"]
            path = out / rel
            if not path.is_file():
                continue
            for bbox in fx.get("bboxes") or []:
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    gt_widths_hires.append(_bbox_width(list(bbox)))

            if rel in seen_files:
                continue
            seen_files.add(rel)

            raw = path.read_bytes()
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            sh, sw = frame.shape[:2]
            source_widths.append(int(sw))
            frame_infer, scale_x, scale_y = maybe_resize_for_infer(frame)
            ih, iw = frame_infer.shape[:2]
            bridge_widths.append(int(iw))
            # GT boxes are in source/hires coords; approximate width on infer plane.
            inv_x = 1.0 / scale_x if scale_x else 1.0
            for bbox in fx.get("bboxes") or []:
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    gt_widths_infer.append(_bbox_width(list(bbox)) * inv_x)

            if live and suite in ("vehicles", "objects"):
                jpeg = encode_jpeg(frame_infer)
                if jpeg is None:
                    continue
                if suite == "vehicles":
                    url = "http://127.0.0.1:8080/vehicle-attribute-recognition"
                else:
                    url = "http://127.0.0.1:8082/object-detection"
                try:
                    data = _post_image(url, jpeg, timeout)
                    size = _decode_result_image_size(data)
                    live_result_sizes.append(
                        {
                            "suite": suite,
                            "file": rel,
                            "bridge_wh": [iw, ih],
                            "result_image_wh": list(size) if size else None,
                        }
                    )
                except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                    live_result_sizes.append(
                        {
                            "suite": suite,
                            "file": rel,
                            "bridge_wh": [iw, ih],
                            "error": str(exc),
                        }
                    )

    bw_sorted = sorted(float(w) for w in bridge_widths)
    bins = [0, 160, 320, 480, 640, 800, 960, 1280, 1920, 10000]
    proposal = int(max(bw_sorted)) if bw_sorted else int(BRIDGE_MAX_WIDTH)
    # Largest round tile size at or below measured max bridge infer width.
    proposal_round = min(proposal, int(BRIDGE_MAX_WIDTH))
    for candidate in (640, 800, 960):
        if candidate <= proposal:
            proposal_round = candidate

    report = {
        "bridge_max_width_env_default": BRIDGE_MAX_WIDTH,
        "n_unique_images": len(seen_files),
        "source_width": {
            "min": min(source_widths) if source_widths else None,
            "max": max(source_widths) if source_widths else None,
            "p50": _percentile(sorted(float(w) for w in source_widths), 50),
        },
        "bridge_infer_width": {
            "min": min(bridge_widths) if bridge_widths else None,
            "max": max(bridge_widths) if bridge_widths else None,
            "p50": _percentile(bw_sorted, 50),
            "p90": _percentile(bw_sorted, 90),
            "histogram": _histogram([float(w) for w in bridge_widths], bins),
        },
        "gt_bbox_width_hires": {
            "n": len(gt_widths_hires),
            "p50": _percentile(sorted(gt_widths_hires), 50),
            "p90": _percentile(sorted(gt_widths_hires), 90),
            "histogram": _histogram(gt_widths_hires, bins),
        },
        "gt_bbox_width_infer_plane": {
            "n": len(gt_widths_infer),
            "p50": _percentile(sorted(gt_widths_infer), 50),
            "p90": _percentile(sorted(gt_widths_infer), 90),
            "histogram": _histogram(gt_widths_infer, bins),
        },
        "live_result_image_sizes": live_result_sizes if live else [],
        "INFER_SLICE_WH_proposal": proposal_round,
        "notes": [
            "Proposal is the largest round size <= measured bridge infer width "
            f"(BRIDGE_MAX_WIDTH={BRIDGE_MAX_WIDTH}).",
            "PaddleX may letterbox further inside the model; result.image size "
            "(if present under --live) is visualization output, not a guarantee "
            "of internal tensor size — treat as optional signal only.",
            "PR2 should set INFER_SLICE_WH from this proposal; do not raise "
            "BRIDGE_MAX_WIDTH in PR1.",
        ],
    }
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="imagenes_muestra")
    parser.add_argument(
        "--report",
        default=str(_SCRIPTS / "infer_slice_measure.json"),
        help="JSON report path (not committed by default)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="POST one sample path through live vehicles/objects (optional)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    try:
        report = measure(out, live=bool(args.live), timeout=float(args.timeout))
    except FileNotFoundError as exc:
        print(f"HARD ERROR: {exc}", file=sys.stderr)
        return 2

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(
        f"INFER_SLICE_WH_proposal={report['INFER_SLICE_WH_proposal']} "
        f"(wrote {report_path})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
