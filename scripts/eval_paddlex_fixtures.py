#!/usr/bin/env python3
"""Evaluate PaddleX fixtures against live localhost ports.

Layers: Core (default gate), Extended + Experimental via --packs all.
HTTP payload rules (mirror benchmark_paddlex):
  - OCR: {"file": b64, "fileType": 1}
  - open_vocab: {"image": b64, "prompt": ...}
  - others: {"image": b64}

Scoring reuses detection/* normalizers (PYTHONPATH=.) and IoU matching.
Tiers: A quantitative GT; B bbox/schema (attrs ignored); C smoke only.

Exit codes:
  0 — pass (Core thresholds + no Core baseline regression; optional layer bars)
  1 — soft failure / threshold or baseline regression
  2 — hard error (missing fixtures, HTTP, config)
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from _eval_packs import (  # noqa: E402
    CORE_PACKS,
    EXPERIMENTAL_PACKS,
    EXTENDED_PACKS,
    PACKS,
    SEED,
    resolve_pack_names,
)

# suite → (url_env, default_url, path_env, default_path, payload_mode)
# payload_mode: image | file | open_vocab
ALL_TARGETS: dict[str, tuple[str, str, str, str, str]] = {
    "objects": (
        "PADDLEX_OBJECTS_URL",
        "http://127.0.0.1:8082",
        "PADDLEX_OBJECTS_PREDICT_PATH",
        "/object-detection",
        "image",
    ),
    "vehicles": (
        "PADDLEX_URL",
        "http://127.0.0.1:8080",
        "PADDLEX_PREDICT_PATH",
        "/vehicle-attribute-recognition",
        "image",
    ),
    "ocr_plates": (
        "PADDLEX_OCR_URL",
        "http://127.0.0.1:8081",
        "PADDLEX_OCR_PREDICT_PATH",
        "/ocr",
        "file",
    ),
    "ocr_text": (
        "PADDLEX_OCR_URL",
        "http://127.0.0.1:8081",
        "PADDLEX_OCR_PREDICT_PATH",
        "/ocr",
        "file",
    ),
    "faces": (
        "PADDLEX_FACES_URL",
        "http://127.0.0.1:8083",
        "PADDLEX_FACES_PREDICT_PATH",
        "/object-detection",
        "image",
    ),
    "pedestrians": (
        "PADDLEX_PEDESTRIANS_URL",
        "http://127.0.0.1:8084",
        "PADDLEX_PEDESTRIANS_PREDICT_PATH",
        "/pedestrian-attribute-recognition",
        "image",
    ),
    "scene": (
        "PADDLEX_SCENE_URL",
        "http://127.0.0.1:8085",
        "PADDLEX_SCENE_PREDICT_PATH",
        "/semantic-segmentation",
        "image",
    ),
    "pose": (
        "PADDLEX_POSE_URL",
        "http://127.0.0.1:8086",
        "PADDLEX_POSE_PREDICT_PATH",
        "/human-keypoint-detection",
        "image",
    ),
    "face_id": (
        "PADDLEX_FACE_ID_URL",
        "http://127.0.0.1:8087",
        "PADDLEX_FACE_ID_PREDICT_PATH",
        "/face-recognition-infer",
        "image",
    ),
    "signs": (
        "PADDLEX_SIGNS_URL",
        "http://127.0.0.1:8088",
        "PADDLEX_SIGNS_PREDICT_PATH",
        "/object-detection",
        "image",
    ),
    "scene_cls": (
        "PADDLEX_SCENE_CLS_URL",
        "http://127.0.0.1:8089",
        "PADDLEX_SCENE_CLS_PREDICT_PATH",
        "/image-classification",
        "image",
    ),
    "instances": (
        "PADDLEX_INSTANCES_URL",
        "http://127.0.0.1:8090",
        "PADDLEX_INSTANCES_PREDICT_PATH",
        "/instance-segmentation",
        "image",
    ),
    "small_objects": (
        "PADDLEX_SMALL_OBJECTS_URL",
        "http://127.0.0.1:8091",
        "PADDLEX_SMALL_OBJECTS_PREDICT_PATH",
        "/small-object-detection",
        "image",
    ),
    "anomaly": (
        "PADDLEX_ANOMALY_URL",
        "http://127.0.0.1:8092",
        "PADDLEX_ANOMALY_PREDICT_PATH",
        "/image-anomaly-detection",
        "image",
    ),
    "open_vocab": (
        "PADDLEX_OPEN_VOCAB_URL",
        "http://127.0.0.1:8093",
        "PADDLEX_OPEN_VOCAB_PREDICT_PATH",
        "/open-vocabulary-detection",
        "open_vocab",
    ),
}

# Back-compat alias used by older Core-only callers/tests.
CORE_TARGETS = {k: ALL_TARGETS[k] for k in CORE_PACKS if k in ALL_TARGETS}


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def normalize_ocr_text(text: Any) -> str:
    """Uppercase alphanumerics only — matches detection.plates.parse_plate spirit."""
    return re.sub(r"[^A-Z0-9]", "", str(text or "").upper())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def match_boxes_by_iou(
    gt_boxes: list[list[float]],
    pred_boxes: list[list[float]],
    iou_fn,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy IoU match. Returns (tp, n_gt, n_pred)."""
    used_pred: set[int] = set()
    tp = 0
    for gb in gt_boxes:
        best_j, best = -1, 0.0
        for j, pb in enumerate(pred_boxes):
            if j in used_pred:
                continue
            score = float(iou_fn(gb, pb))
            if score > best:
                best, best_j = score, j
        if best_j >= 0 and best >= iou_threshold:
            used_pred.add(best_j)
            tp += 1
    return tp, len(gt_boxes), len(pred_boxes)


def match_labeled_boxes(
    gt_boxes: list[list[float]],
    gt_labels: list[str],
    pred_boxes: list[list[float]],
    pred_labels: list[str],
    iou_fn,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Match requiring IoU + case-insensitive label equality."""
    used_pred: set[int] = set()
    tp = 0
    for gb, gl in zip(gt_boxes, gt_labels):
        gl_n = str(gl).strip().lower()
        best_j, best = -1, 0.0
        for j, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
            if j in used_pred:
                continue
            if str(pl).strip().lower() != gl_n:
                continue
            score = float(iou_fn(gb, pb))
            if score > best:
                best, best_j = score, j
        if best_j >= 0 and best >= iou_threshold:
            used_pred.add(best_j)
            tp += 1
    return tp, len(gt_boxes), len(pred_boxes)


def rate(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def bbox_schema_ok(det: dict[str, Any]) -> bool:
    """Valid bbox + label/entity_type — Tier B attrs ignored."""
    bbox = det.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return False
    try:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
    except (TypeError, ValueError):
        return False
    if x2 <= x1 or y2 <= y1:
        return False
    if "label" not in det and "entity_type" not in det:
        return False
    return True


def vehicle_schema_ok(det: dict[str, Any]) -> bool:
    """Tier B vehicles: valid bbox + required keys; attributes ignored."""
    return bbox_schema_ok(det)


def compare_to_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    """Return breach messages. Core always checked; extended/experimental if present."""
    breaches: list[str] = []
    for section in ("core", "extended", "experimental"):
        bars_root = thresholds.get(section) or {}
        for suite, bars in bars_root.items():
            if suite not in metrics:
                continue
            m = metrics[suite]
            for key, bar_key in (
                ("recall", "min_recall"),
                ("precision", "min_precision"),
                ("bbox_match_rate", "min_bbox_match_rate"),
                ("schema_ok_rate", "min_schema_ok_rate"),
                ("exact_match_rate", "min_exact_match_rate"),
                ("smoke_ok_rate", "min_smoke_ok_rate"),
            ):
                if bar_key not in bars:
                    continue
                if m.get(key, 0) < float(bars[bar_key]):
                    breaches.append(
                        f"{suite}.{key} {m.get(key)} < {bar_key} {bars[bar_key]}"
                    )
    return breaches


def compare_to_baseline(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    """Regression vs committed baseline (skip null baseline metrics)."""
    max_reg = thresholds.get("baseline_max_regression") or {}
    base_m = baseline.get("metrics") or {}
    regressions: list[str] = []

    def _check(suite: str, key: str, reg_key: str) -> None:
        cur = metrics.get(suite, {}).get(key)
        ref = (base_m.get(suite) or {}).get(key)
        if cur is None or ref is None:
            return
        allowed = float(max_reg.get(reg_key, 0.0))
        if float(cur) + 1e-9 < float(ref) - allowed:
            regressions.append(
                f"{suite}.{key} {cur} < baseline {ref} − {allowed}"
            )

    # Core gate keys
    _check("objects", "recall", "objects_recall")
    _check("objects", "precision", "objects_precision")
    _check("vehicles", "bbox_match_rate", "vehicles_bbox_match_rate")
    _check("vehicles", "schema_ok_rate", "vehicles_schema_ok_rate")
    _check("ocr_plates", "exact_match_rate", "ocr_plates_exact_match_rate")
    # Extended / Experimental opt-in (null baseline → skip)
    for suite in list(EXTENDED_PACKS) + list(EXPERIMENTAL_PACKS):
        for key in (
            "recall",
            "precision",
            "bbox_match_rate",
            "schema_ok_rate",
            "exact_match_rate",
            "smoke_ok_rate",
        ):
            _check(suite, key, f"{suite}_{key}")
    return regressions


# ---------------------------------------------------------------------------
# HTTP + normalizers
# ---------------------------------------------------------------------------


def _read_jpeg(path: Path, *, via_bridge_preprocess: bool = False) -> bytes:
    """Load fixture JPEG; optionally apply bridge maybe_resize_for_infer (no tiles)."""
    raw = path.read_bytes()
    if not via_bridge_preprocess:
        return raw
    import cv2
    import numpy as np

    from detection.common.geometry import encode_jpeg, maybe_resize_for_infer

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return raw
    frame_infer, _sx, _sy = maybe_resize_for_infer(frame)
    encoded = encode_jpeg(frame_infer)
    return encoded if encoded is not None else raw


def _load_bgr_frame(path: Path):
    """Decode fixture to BGR ndarray for tiled sync harness."""
    import cv2
    import numpy as np

    arr = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"cannot decode fixture image: {path}")
    return frame


def _detections_via_tiled_sync(suite: str, path: Path) -> list[dict[str, Any]]:
    """Host-side call to infer_*_tiled_sync (same núcleo as bridge; no asyncio)."""
    frame = _load_bgr_frame(path)
    if suite == "vehicles":
        from detection.vehicles.client import infer_vehicles_tiled_sync

        dets = infer_vehicles_tiled_sync(frame)
    elif suite == "objects":
        from detection.objects.client import infer_objects_tiled_sync

        dets = infer_objects_tiled_sync(frame)
    else:
        raise RuntimeError(f"--via-tiled-sync only supports vehicles/objects, got {suite}")
    if dets is None:
        raise RuntimeError(f"tiled sync failed for {suite} ({path.name})")
    return dets


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = int(getattr(resp, "status", 200) or 200)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} for {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"connection error for {url}: {exc.reason}") from exc
    if status >= 400:
        raise RuntimeError(f"HTTP {status} for {url}")
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"non-object JSON from {url}")
    return data


def _predict(
    suite: str,
    jpeg: bytes,
    timeout: float,
) -> dict[str, Any]:
    if suite not in ALL_TARGETS:
        raise RuntimeError(f"no TARGETS entry for suite {suite}")
    url_env, url_def, path_env, path_def, mode = ALL_TARGETS[suite]
    base = os.getenv(url_env, url_def)
    path = os.getenv(path_env, path_def)
    url = f"{base.rstrip('/')}{path}"
    b64 = base64.b64encode(jpeg).decode("ascii")
    if mode == "file":
        payload: dict[str, Any] = {"file": b64, "fileType": 1}
    elif mode == "open_vocab":
        prompt = os.getenv("OPEN_VOCAB_PROMPT", "person,car,traffic sign")
        payload = {"image": b64, "prompt": prompt}
    else:
        payload = {"image": b64}
        if suite in {"scene_cls", "anomaly", "face_id", "faces"}:
            payload["visualize"] = False
    return _post_json(url, payload, timeout)


def _normalize_detections(suite: str, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return list of detection dicts (wrap single-entity normalizers)."""
    if suite == "objects":
        from detection.objects.client import normalize_object_detection_result

        return normalize_object_detection_result(data)
    if suite == "vehicles":
        from detection.vehicles.client import normalize_vehicle_result

        return normalize_vehicle_result(data)
    if suite == "signs":
        from detection.signs.client import normalize_signs_result

        return normalize_signs_result(data)
    if suite == "faces":
        from detection.faces.client import normalize_face_result

        return normalize_face_result(data)
    if suite == "pose":
        from detection.pose.client import normalize_pose_result

        return normalize_pose_result(data)
    if suite == "pedestrians":
        from detection.pedestrians.client import normalize_pedestrian_result

        return normalize_pedestrian_result(data)
    if suite == "scene":
        from detection.scene.client import normalize_scene_result

        one = normalize_scene_result(data)
        return [one] if one else []
    if suite == "instances":
        from detection.instances.client import normalize_instances_result

        return normalize_instances_result(data)
    if suite == "small_objects":
        from detection.small_objects.client import normalize_small_objects_result

        return normalize_small_objects_result(data)
    if suite == "open_vocab":
        from detection.open_vocab.client import normalize_open_vocab_result

        return normalize_open_vocab_result(data)
    if suite == "scene_cls":
        from detection.scene_cls.client import normalize_scene_cls_result

        one = normalize_scene_cls_result(data)
        return [one] if one else []
    if suite == "anomaly":
        from detection.anomaly.client import normalize_anomaly_result

        one = normalize_anomaly_result(data)
        return [one] if one else []
    if suite == "face_id":
        from detection.face_id.client import normalize_face_id_result

        return normalize_face_id_result(data)
    return []


def _extract_ocr_texts(data: dict[str, Any]) -> list[str]:
    result = data.get("result", data) if isinstance(data, dict) else {}
    texts: list[str] = []
    if not isinstance(result, dict):
        return texts
    ocr_results = result.get("ocrResults") or []
    for item in ocr_results:
        if not isinstance(item, dict):
            continue
        pruned = item.get("prunedResult") or {}
        for t in pruned.get("rec_texts") or []:
            texts.append(str(t))
    if not texts:
        for t in result.get("rec_texts") or []:
            texts.append(str(t))
    # Scene OCR shape via text normalizer
    if not texts:
        try:
            from detection.text.client import normalize_scene_ocr_result

            for d in normalize_scene_ocr_result(data):
                if d.get("text"):
                    texts.append(str(d["text"]))
        except Exception:  # noqa: BLE001
            pass
    return texts


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "PyYAML required. Install: python -m pip install -r scripts/requirements-eval.txt"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Invalid thresholds YAML: {path}")
    return data


def _load_gt(out: Path, suite: str) -> dict[str, Any]:
    path = out / "gt" / f"{suite}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing GT: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_manifest_hashes(out: Path, suites: list[str]) -> list[str]:
    """Verify hashes for suites under evaluation (partial manifests OK for core-only)."""
    manifest_path = out / "gt" / "manifest.json"
    if not manifest_path.is_file():
        return [f"missing manifest: {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if int(manifest.get("seed", -1)) != SEED:
        errors.append(f"manifest seed {manifest.get('seed')} != {SEED}")
    suite_blocks = manifest.get("suites") or {}
    for suite in suites:
        block = suite_blocks.get(suite)
        if not block:
            errors.append(f"manifest missing suite {suite}")
            continue
        for img in block.get("images") or []:
            path = out / img["file"]
            if not path.is_file():
                errors.append(f"missing file {img['file']}")
                continue
            got = sha256_file(path)
            if got != img.get("sha256"):
                errors.append(
                    f"hash mismatch {img['file']}: {got} != {img.get('sha256')}"
                )
    return errors


def _write_failure(
    failures_root: Path,
    suite: str,
    fixture_id: str,
    file_name: str,
    reason: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    dest_dir = failures_root / suite
    dest_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "suite": suite,
        "id": fixture_id,
        "file": file_name,
        "reason": reason,
    }
    if extra:
        payload["extra"] = extra
    (dest_dir / f"{fixture_id}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Suite scorers
# ---------------------------------------------------------------------------


def score_detection_tier_a(
    suite: str,
    out: Path,
    gt: dict[str, Any],
    iou_threshold: float,
    timeout: float,
    failures_root: Path,
    *,
    require_labels: bool = True,
    via_bridge_preprocess: bool = False,
    via_tiled_sync: bool = False,
) -> dict[str, Any]:
    from detection.common.tracking import iou

    tp_total = fp_like = gt_total = 0
    n_ok = 0
    fixtures = gt.get("fixtures") or []
    for fx in fixtures:
        path = out / fx["file"]
        try:
            if via_tiled_sync:
                dets = _detections_via_tiled_sync(suite, path)
            else:
                jpeg = _read_jpeg(path, via_bridge_preprocess=via_bridge_preprocess)
                data = _predict(suite, jpeg, timeout)
                dets = _normalize_detections(suite, data)
        except Exception as exc:  # noqa: BLE001
            _write_failure(
                failures_root, suite, fx["id"], fx["file"], f"predict_error: {exc}"
            )
            gt_total += len(fx.get("bboxes") or [])
            continue
        gt_boxes = list(fx.get("bboxes") or [])
        gt_labels = list(fx.get("labels") or [])
        pred_boxes = [d["bbox"] for d in dets if d.get("bbox")]
        pred_labels = [str(d.get("label") or "") for d in dets if d.get("bbox")]
        if require_labels and gt_labels:
            tp, n_gt, n_pred = match_labeled_boxes(
                gt_boxes, gt_labels, pred_boxes, pred_labels, iou, iou_threshold
            )
        else:
            tp, n_gt, n_pred = match_boxes_by_iou(
                gt_boxes, pred_boxes, iou, iou_threshold
            )
        tp_total += tp
        gt_total += n_gt
        fp_like += max(0, n_pred - tp)
        if tp < n_gt:
            _write_failure(
                failures_root,
                suite,
                fx["id"],
                fx["file"],
                f"tier_a_miss tp={tp} gt={n_gt} pred={n_pred}",
                {"gt_labels": gt_labels, "pred_labels": pred_labels},
            )
        else:
            n_ok += 1
    recall = rate(tp_total, gt_total)
    precision = rate(tp_total, tp_total + fp_like)
    return {
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "n_fixtures": len(fixtures),
        "tp": tp_total,
        "gt_boxes": gt_total,
        "fixtures_ok": n_ok,
    }


def score_detection_tier_b(
    suite: str,
    out: Path,
    gt: dict[str, Any],
    iou_threshold: float,
    timeout: float,
    failures_root: Path,
    *,
    via_bridge_preprocess: bool = False,
    via_tiled_sync: bool = False,
) -> dict[str, Any]:
    """Bbox + schema; attributes / class labels ignored."""
    from detection.common.tracking import iou

    match_tp = gt_total = 0
    schema_ok = schema_n = 0
    fixtures = gt.get("fixtures") or []
    for fx in fixtures:
        path = out / fx["file"]
        try:
            if via_tiled_sync:
                dets = _detections_via_tiled_sync(suite, path)
            else:
                jpeg = _read_jpeg(path, via_bridge_preprocess=via_bridge_preprocess)
                data = _predict(suite, jpeg, timeout)
                dets = _normalize_detections(suite, data)
        except Exception as exc:  # noqa: BLE001
            _write_failure(
                failures_root, suite, fx["id"], fx["file"], f"predict_error: {exc}"
            )
            gt_total += len(fx.get("bboxes") or [])
            continue
        for d in dets:
            schema_n += 1
            if bbox_schema_ok(d):
                schema_ok += 1
            else:
                _write_failure(
                    failures_root,
                    suite,
                    fx["id"],
                    fx["file"],
                    "schema_invalid",
                    {"det": d},
                )
        gt_boxes = list(fx.get("bboxes") or [])
        pred_boxes = [d["bbox"] for d in dets if bbox_schema_ok(d)]
        if gt_boxes:
            tp, n_gt, _ = match_boxes_by_iou(gt_boxes, pred_boxes, iou, iou_threshold)
            match_tp += tp
            gt_total += n_gt
            if tp < n_gt:
                _write_failure(
                    failures_root,
                    suite,
                    fx["id"],
                    fx["file"],
                    f"bbox_miss tp={tp} gt={n_gt}",
                )
        else:
            # No GT boxes (e.g. scene / scene_cls) — schema-only honesty.
            if not dets:
                _write_failure(
                    failures_root,
                    suite,
                    fx["id"],
                    fx["file"],
                    "schema_empty_response",
                )
    return {
        "bbox_match_rate": round(rate(match_tp, gt_total), 4) if gt_total else 1.0,
        "schema_ok_rate": round(rate(schema_ok, schema_n) if schema_n else 0.0, 4),
        "n_fixtures": len(fixtures),
        "tp": match_tp,
        "gt_boxes": gt_total,
    }


def score_ocr_text_suite(
    suite: str,
    out: Path,
    gt: dict[str, Any],
    timeout: float,
    failures_root: Path,
    *,
    via_bridge_preprocess: bool = False,
) -> dict[str, Any]:
    fixtures = gt.get("fixtures") or []
    hits = 0
    for fx in fixtures:
        path = out / fx["file"]
        jpeg = _read_jpeg(path, via_bridge_preprocess=via_bridge_preprocess)
        expect = normalize_ocr_text(fx.get("text"))
        try:
            data = _predict(suite, jpeg, timeout)
            texts = [normalize_ocr_text(t) for t in _extract_ocr_texts(data)]
        except Exception as exc:  # noqa: BLE001
            _write_failure(
                failures_root, suite, fx["id"], fx["file"], f"predict_error: {exc}"
            )
            continue
        if expect and any(expect in t or t in expect for t in texts if t):
            hits += 1
        else:
            _write_failure(
                failures_root,
                suite,
                fx["id"],
                fx["file"],
                f"text_miss expect={expect} got={texts}",
            )
    return {
        "exact_match_rate": round(rate(hits, len(fixtures)), 4),
        "n_fixtures": len(fixtures),
        "hits": hits,
    }


def score_tier_c_smoke(
    suite: str,
    out: Path,
    gt: dict[str, Any],
    timeout: float,
    failures_root: Path,
    *,
    via_bridge_preprocess: bool = False,
) -> dict[str, Any]:
    """Tier C: HTTP + normalize succeeds (no accuracy claim)."""
    fixtures = gt.get("fixtures") or []
    ok = 0
    for fx in fixtures:
        path = out / fx["file"]
        jpeg = _read_jpeg(path, via_bridge_preprocess=via_bridge_preprocess)
        try:
            data = _predict(suite, jpeg, timeout)
            dets = _normalize_detections(suite, data)
            if dets is None:
                raise RuntimeError("normalize returned None")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            _write_failure(
                failures_root, suite, fx["id"], fx["file"], f"smoke_fail: {exc}"
            )
    return {
        "smoke_ok_rate": round(rate(ok, len(fixtures)), 4),
        "n_fixtures": len(fixtures),
        "ok": ok,
        "tier": "C",
    }


def score_objects(
    out: Path,
    gt: dict[str, Any],
    iou_threshold: float,
    timeout: float,
    failures_root: Path,
) -> dict[str, Any]:
    return score_detection_tier_a(
        "objects", out, gt, iou_threshold, timeout, failures_root
    )


def score_vehicles(
    out: Path,
    gt: dict[str, Any],
    iou_threshold: float,
    timeout: float,
    failures_root: Path,
) -> dict[str, Any]:
    return score_detection_tier_b(
        "vehicles", out, gt, iou_threshold, timeout, failures_root
    )


def score_ocr_plates(
    out: Path,
    gt: dict[str, Any],
    timeout: float,
    failures_root: Path,
) -> dict[str, Any]:
    return score_ocr_text_suite("ocr_plates", out, gt, timeout, failures_root)


def score_suite(
    suite: str,
    out: Path,
    gt: dict[str, Any],
    iou_threshold: float,
    timeout: float,
    failures_root: Path,
    *,
    via_bridge_preprocess: bool = False,
    via_tiled_sync: bool = False,
) -> dict[str, Any]:
    meta = PACKS.get(suite) or {}
    tier = str(gt.get("tier") or meta.get("tier") or "A").upper()
    if suite in ("ocr_plates", "ocr_text"):
        return score_ocr_text_suite(
            suite,
            out,
            gt,
            timeout,
            failures_root,
            via_bridge_preprocess=via_bridge_preprocess,
        )
    if tier == "C":
        return score_tier_c_smoke(
            suite,
            out,
            gt,
            timeout,
            failures_root,
            via_bridge_preprocess=via_bridge_preprocess,
        )
    if tier == "B":
        return score_detection_tier_b(
            suite,
            out,
            gt,
            iou_threshold,
            timeout,
            failures_root,
            via_bridge_preprocess=via_bridge_preprocess,
            via_tiled_sync=via_tiled_sync,
        )
    # Tier A detection — faces/pose may use loose labels (face / person_pose)
    require_labels = suite not in ("faces", "pose")
    return score_detection_tier_a(
        suite,
        out,
        gt,
        iou_threshold,
        timeout,
        failures_root,
        require_labels=require_labels,
        via_bridge_preprocess=via_bridge_preprocess,
        via_tiled_sync=via_tiled_sync,
    )


def run_eval(
    out: Path,
    suites: list[str],
    thresholds_path: Path,
    baseline_path: Path,
    report_path: Path,
    timeout: float,
    *,
    packs_arg: str = "core",
    via_bridge_preprocess: bool = False,
    via_tiled_sync: bool = False,
) -> int:
    thresholds = _load_yaml(thresholds_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    iou_threshold = float(thresholds.get("match_iou", 0.5))

    hash_errors = _verify_manifest_hashes(out, suites)
    if hash_errors:
        print("HARD ERROR: manifest/hash issues:", file=sys.stderr)
        for e in hash_errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    failures_root = out / "failures"
    failures_root.mkdir(parents=True, exist_ok=True)
    for suite in suites:
        sdir = failures_root / suite
        if sdir.is_dir():
            for old in sdir.glob("*.json"):
                old.unlink()

    metrics: dict[str, Any] = {}
    hard_error: Optional[str] = None
    for suite in suites:
        if suite not in ALL_TARGETS:
            print(f"SKIP unknown target suite: {suite}", file=sys.stderr)
            continue
        try:
            gt = _load_gt(out, suite)
        except FileNotFoundError as exc:
            # Core missing GT is hard; Extended/Experimental missing is skip when packs=all
            if suite in CORE_PACKS:
                hard_error = str(exc)
                break
            print(f"SKIP missing GT for opt-in suite {suite}: {exc}", file=sys.stderr)
            continue
        tier = gt.get("tier") or PACKS.get(suite, {}).get("tier")
        print(f"==> eval {suite} (tier={tier})")
        metrics[suite] = score_suite(
            suite,
            out,
            gt,
            iou_threshold,
            timeout,
            failures_root,
            via_bridge_preprocess=via_bridge_preprocess,
            via_tiled_sync=via_tiled_sync,
        )

    if hard_error:
        print(f"HARD ERROR: {hard_error}", file=sys.stderr)
        return 2

    # Core gate always; extended/experimental bars only for suites that ran.
    breaches = compare_to_thresholds(metrics, thresholds)
    regressions = compare_to_baseline(metrics, baseline, thresholds)

    # When --packs core, ignore non-core breach noise (shouldn't exist).
    if packs_arg == "core":
        breaches = [b for b in breaches if b.split(".", 1)[0] in CORE_PACKS]
        regressions = [r for r in regressions if r.split(".", 1)[0] in CORE_PACKS]

    report = {
        "seed": SEED,
        "packs": packs_arg,
        "suites": suites,
        "via_bridge_preprocess": via_bridge_preprocess,
        "via_tiled_sync": via_tiled_sync,
        "metrics": metrics,
        "threshold_breaches": breaches,
        "baseline_regressions": regressions,
        "baseline_path": str(baseline_path),
        "thresholds_path": str(thresholds_path),
        "pass": not breaches and not regressions,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    if breaches or regressions:
        print("FAIL: threshold/baseline gate", file=sys.stderr)
        for b in breaches + regressions:
            print(f"  - {b}", file=sys.stderr)
        return 1
    gate = "Core" if packs_arg == "core" else "multi-layer"
    print(f"PASS: {gate} accuracy gate")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packs", default="core", choices=("core", "all"))
    parser.add_argument(
        "--only",
        "--pipelines",
        dest="only",
        help="Comma-separated suite / pipeline subset (alias: --pipelines)",
    )
    parser.add_argument("--out", default="imagenes_muestra")
    parser.add_argument(
        "--thresholds",
        default=str(_SCRIPTS / "eval_thresholds.yaml"),
    )
    parser.add_argument(
        "--baseline",
        default=str(_SCRIPTS / "eval_baseline.json"),
    )
    parser.add_argument(
        "--report",
        default=str(_SCRIPTS / "eval_report.json"),
        help="Where to write the JSON report (not committed)",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--via-bridge-preprocess",
        action="store_true",
        help=(
            "Apply detection.common.geometry.maybe_resize_for_infer before POST "
            "(PR1 baseline; no tiling). Uses current BRIDGE_MAX_WIDTH default."
        ),
    )
    parser.add_argument(
        "--via-tiled-sync",
        action="store_true",
        help=(
            "vehicles/objects: call infer_*_tiled_sync (same núcleo as bridge). "
            "Mutually exclusive with --via-bridge-preprocess for those suites."
        ),
    )
    args = parser.parse_args(argv)

    out = Path(args.out).resolve()
    try:
        suites = resolve_pack_names(args.packs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    only = {s.strip() for s in (args.only or "").split(",") if s.strip()}
    if only:
        suites = [s for s in suites if s in only]
    # Default core never pulls Extended/Experimental even if --only names them
    # unless --packs all (or only intersects with resolved list from packs).
    if args.packs == "core":
        suites = [s for s in suites if s in CORE_PACKS]
    if not suites:
        print("No suites selected", file=sys.stderr)
        return 2

    return run_eval(
        out=out,
        suites=suites,
        thresholds_path=Path(args.thresholds),
        baseline_path=Path(args.baseline),
        report_path=Path(args.report),
        timeout=args.timeout,
        packs_arg=args.packs,
        via_bridge_preprocess=bool(args.via_bridge_preprocess),
        via_tiled_sync=bool(args.via_tiled_sync),
    )


if __name__ == "__main__":
    raise SystemExit(main())
