"""Unit tests for PaddleX eval harness matchers / threshold helpers.

No live FiftyOne or PaddleX. Run from repo root:

    PYTHONPATH=. python3 tests/test_eval_match.py
    pytest tests/test_eval_match.py -q
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO / "scripts"
for p in (_REPO, _SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from detection.common.tracking import iou  # noqa: E402

from eval_paddlex_fixtures import (  # noqa: E402
    ALL_TARGETS,
    bbox_schema_ok,
    compare_to_baseline,
    compare_to_thresholds,
    match_boxes_by_iou,
    match_labeled_boxes,
    normalize_ocr_text,
    sha256_file,
    vehicle_schema_ok,
)
from _eval_packs import (  # noqa: E402
    ALL_PACKS,
    CORE_PACKS,
    EXPERIMENTAL_PACKS,
    EXTENDED_PACKS,
    PACKS,
    SEED,
    resolve_pack_names,
)


class NormalizeOcrTextTests(unittest.TestCase):
    def test_strips_non_alnum_and_uppercases(self) -> None:
        self.assertEqual(normalize_ocr_text("ab-c 12"), "ABC12")
        self.assertEqual(normalize_ocr_text(None), "")


class MatchBoxesTests(unittest.TestCase):
    def test_iou_perfect_overlap(self) -> None:
        a = [0.0, 0.0, 10.0, 10.0]
        self.assertAlmostEqual(iou(a, a), 1.0)

    def test_match_boxes_counts_tp(self) -> None:
        gt = [[0, 0, 10, 10], [20, 20, 30, 30]]
        pred = [[1, 1, 9, 9], [100, 100, 110, 110]]
        tp, n_gt, n_pred = match_boxes_by_iou(gt, pred, iou, 0.3)
        self.assertEqual(tp, 1)
        self.assertEqual(n_gt, 2)
        self.assertEqual(n_pred, 2)

    def test_labeled_match_requires_label(self) -> None:
        gt_b = [[0, 0, 10, 10]]
        gt_l = ["person"]
        pred_b = [[0, 0, 10, 10]]
        pred_l = ["car"]
        tp, _, _ = match_labeled_boxes(gt_b, gt_l, pred_b, pred_l, iou, 0.5)
        self.assertEqual(tp, 0)
        tp2, _, _ = match_labeled_boxes(gt_b, gt_l, pred_b, ["person"], iou, 0.5)
        self.assertEqual(tp2, 1)


class ThresholdBaselineTests(unittest.TestCase):
    def test_threshold_breach(self) -> None:
        thresholds = {
            "core": {
                "objects": {"min_recall": 0.8, "min_precision": 0.8},
                "vehicles": {"min_bbox_match_rate": 0.5, "min_schema_ok_rate": 0.9},
                "ocr_plates": {"min_exact_match_rate": 0.9},
            }
        }
        metrics = {
            "objects": {"recall": 0.5, "precision": 0.9},
            "vehicles": {"bbox_match_rate": 0.7, "schema_ok_rate": 1.0},
            "ocr_plates": {"exact_match_rate": 1.0},
        }
        breaches = compare_to_thresholds(metrics, thresholds)
        self.assertTrue(any("objects.recall" in b for b in breaches))

    def test_baseline_skips_null_and_flags_regression(self) -> None:
        thresholds = {
            "baseline_max_regression": {
                "objects_recall": 0.05,
                "objects_precision": 0.05,
                "vehicles_bbox_match_rate": 0.05,
                "vehicles_schema_ok_rate": 0.02,
                "ocr_plates_exact_match_rate": 0.05,
            }
        }
        baseline = {
            "metrics": {
                "objects": {"recall": None, "precision": 0.9},
                "vehicles": {"bbox_match_rate": 0.8, "schema_ok_rate": 1.0},
                "ocr_plates": {"exact_match_rate": 0.95},
            }
        }
        metrics = {
            "objects": {"recall": 0.1, "precision": 0.7},
            "vehicles": {"bbox_match_rate": 0.8, "schema_ok_rate": 1.0},
            "ocr_plates": {"exact_match_rate": 0.95},
        }
        regs = compare_to_baseline(metrics, baseline, thresholds)
        # null recall skipped; precision 0.7 < 0.9 - 0.05
        self.assertTrue(any("objects.precision" in r for r in regs))
        self.assertFalse(any("objects.recall" in r for r in regs))


class ManifestHashTests(unittest.TestCase):
    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            data = b"vision-intelligence-eval"
            path.write_bytes(data)
            self.assertEqual(sha256_file(path), hashlib.sha256(data).hexdigest())


class VehicleSchemaTests(unittest.TestCase):
    def test_schema_ok_ignores_attrs(self) -> None:
        self.assertTrue(
            vehicle_schema_ok(
                {"label": "sedan", "bbox": [1, 2, 30, 40], "color": "wrong"}
            )
        )
        self.assertFalse(vehicle_schema_ok({"label": "sedan", "bbox": [10, 10, 5, 5]}))


class PackRegistryTests(unittest.TestCase):
    def test_core_default(self) -> None:
        self.assertEqual(resolve_pack_names("core"), list(CORE_PACKS))
        self.assertEqual(SEED, 51)
        self.assertIn("objects", CORE_PACKS)
        self.assertIn("vehicles", CORE_PACKS)
        self.assertIn("ocr_plates", CORE_PACKS)

    def test_all_includes_extended_and_experimental(self) -> None:
        names = resolve_pack_names("all")
        self.assertEqual(names, list(ALL_PACKS))
        for s in ("signs", "faces", "pose", "pedestrians", "scene", "ocr_text"):
            self.assertIn(s, EXTENDED_PACKS)
            self.assertNotEqual(PACKS[s]["source"], "stub")
        for s in (
            "instances",
            "small_objects",
            "open_vocab",
            "scene_cls",
            "anomaly",
            "face_id",
        ):
            self.assertIn(s, EXPERIMENTAL_PACKS)
            self.assertNotEqual(PACKS[s]["source"], "stub")
        self.assertEqual(PACKS["pedestrians"]["tier"], "B")
        self.assertEqual(PACKS["anomaly"]["tier"], "C")
        self.assertEqual(PACKS["ocr_text"]["source"], "synthetic_text")
        self.assertEqual(PACKS["face_id"]["source"], "synthetic_face_id")

    def test_targets_cover_extended_and_ports_8089_8093(self) -> None:
        for s in EXTENDED_PACKS + EXPERIMENTAL_PACKS:
            self.assertIn(s, ALL_TARGETS)
        self.assertIn(":8089", ALL_TARGETS["scene_cls"][1])
        self.assertIn(":8090", ALL_TARGETS["instances"][1])
        self.assertIn(":8091", ALL_TARGETS["small_objects"][1])
        self.assertIn(":8092", ALL_TARGETS["anomaly"][1])
        self.assertIn(":8093", ALL_TARGETS["open_vocab"][1])


class ExtendedThresholdTests(unittest.TestCase):
    def test_extended_bars_opt_in(self) -> None:
        thresholds = {
            "core": {"objects": {"min_recall": 0.5, "min_precision": 0.5}},
            "extended": {"pedestrians": {"min_schema_ok_rate": 0.9}},
            "experimental": {"anomaly": {"min_smoke_ok_rate": 0.95}},
        }
        metrics = {
            "objects": {"recall": 0.9, "precision": 0.9},
            "pedestrians": {"schema_ok_rate": 0.5},
            "anomaly": {"smoke_ok_rate": 1.0},
        }
        breaches = compare_to_thresholds(metrics, thresholds)
        self.assertTrue(any("pedestrians.schema_ok_rate" in b for b in breaches))
        self.assertFalse(any("anomaly" in b for b in breaches))

    def test_bbox_schema_ok(self) -> None:
        self.assertTrue(
            bbox_schema_ok({"entity_type": "person", "bbox": [0, 0, 10, 10]})
        )
        self.assertFalse(bbox_schema_ok({"bbox": [0, 0, 10, 10]}))


if __name__ == "__main__":
    unittest.main()
