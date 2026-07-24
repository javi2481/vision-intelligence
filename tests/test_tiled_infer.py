"""Tests InferenceSlicer NMS-A + invariante hires (tiling on/off)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from detection.common.geometry import scale_detections
from detection.common.tiled_infer import (
    class_id_for_tile_nms,
    detections_to_vi_raw,
    infer_tiled_sync,
    vi_raw_to_detections,
)


class ClassIdForTileNmsTests(unittest.TestCase):
    def test_stable_within_capability_independent_across(self) -> None:
        a = class_id_for_tile_nms("car", capability="vehicles")
        b = class_id_for_tile_nms("car", capability="vehicles")
        c = class_id_for_tile_nms("truck", capability="vehicles")
        d = class_id_for_tile_nms("car", capability="objects")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        # objects bucket is independent (may reuse 0)
        self.assertEqual(d, class_id_for_tile_nms("car", capability="objects"))


class ViDetectionsRoundTripTests(unittest.TestCase):
    def test_round_trip_preserves_bbox_label_color(self) -> None:
        raw = [
            {
                "label": "sedan",
                "score": 0.9,
                "bbox": [10.0, 20.0, 110.0, 120.0],
                "color": "white",
                "entity_type": "vehicle",
            }
        ]
        dets = vi_raw_to_detections(raw, capability="vehicles")
        self.assertEqual(len(dets), 1)
        self.assertIsNotNone(dets.class_id)
        self.assertIsNotNone(dets.confidence)
        back = detections_to_vi_raw(dets, capability="vehicles")
        self.assertEqual(back[0]["label"], "sedan")
        self.assertEqual(back[0]["color"], "white")
        self.assertEqual(back[0]["bbox"], [10.0, 20.0, 110.0, 120.0])


def _fake_vehicle_box(_data):
    """Same tile-local box for every JPEG (slice or full frame)."""
    return [
        {
            "label": "sedan",
            "score": 0.95,
            "bbox": [10.0, 10.0, 100.0, 80.0],
            "color": "red",
            "entity_type": "vehicle",
        }
    ]


class TilingOnOffSameFixtureTests(unittest.TestCase):
    """Anti double-scale: frame ≤ slice_wh → one tile, offset (0,0).

    Same mock box must match tiling vs non-tiling (tolerance chica). Applying
    scale_detections after tiled hires boxes diverges (regresión del plan).
    """

    def test_single_tile_matches_full_frame_and_rejects_double_scale(self) -> None:
        # Fits in one 640 tile → InferenceSlicer uses offset (0,0); move_detections
        # leaves coords unchanged (supervision 0.28 _run_callback).
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        expected = [10.0, 10.0, 100.0, 80.0]

        with patch(
            "detection.common.tiled_infer.post_image_predict_sync",
            return_value={"errorCode": 0, "result": {}},
        ):
            tiled = infer_tiled_sync(
                frame,
                base_url="http://example.invalid",
                predict_path="/vehicle-attribute-recognition",
                normalize_response=_fake_vehicle_box,
                capability="vehicles",
                slice_wh=640,
                overlap_wh=100,
                thread_workers=1,
            )
        self.assertIsNotNone(tiled)
        assert tiled is not None
        self.assertEqual(len(tiled), 1)
        self.assertEqual(tiled[0]["bbox"], expected)

        # Non-tiling path: same fixture JPEG coords (no bridge downscale on 640).
        non_tiled = _fake_vehicle_box({})
        self.assertEqual(non_tiled[0]["bbox"], expected)
        for a, b in zip(tiled[0]["bbox"], non_tiled[0]["bbox"]):
            self.assertAlmostEqual(a, b, delta=1.0)

        # Double-scale bug: scale after already-hires tiled boxes.
        wrongly = [dict(tiled[0])]
        scale_detections(wrongly, 2.0, 2.0)
        self.assertGreater(abs(wrongly[0]["bbox"][0] - tiled[0]["bbox"][0]), 5.0)


class InferTiledSyncTests(unittest.TestCase):
    def test_slicer_moves_tile_coords_to_hires(self) -> None:
        """Two tiles: right-tile local box must land at x >= stride after move."""
        frame = np.zeros((640, 1280, 3), dtype=np.uint8)
        slice_wh = 640
        overlap_wh = 100
        stride = slice_wh - overlap_wh  # 540 per supervision stride formula

        with patch(
            "detection.common.tiled_infer.post_image_predict_sync",
            return_value={"errorCode": 0, "result": {}},
        ):
            out = infer_tiled_sync(
                frame,
                base_url="http://example.invalid",
                predict_path="/vehicle-attribute-recognition",
                normalize_response=_fake_vehicle_box,
                capability="vehicles",
                slice_wh=slice_wh,
                overlap_wh=overlap_wh,
                thread_workers=1,
            )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreaterEqual(len(out), 1)
        xs = sorted(d["bbox"][0] for d in out)
        # At least one detection from the second tile (offset x = stride).
        self.assertTrue(
            any(x >= stride - 1.0 for x in xs),
            f"expected a moved box with x>={stride}, got xs={xs}",
        )
        for det in out:
            x1, y1, x2, y2 = det["bbox"]
            self.assertLess(x2, 1280 + 1)
            self.assertLess(y2, 640 + 1)
            self.assertGreaterEqual(x1, 0.0)
            self.assertLess(x2 - x1, 200.0)

    def test_all_tile_failures_return_none(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch(
            "detection.common.tiled_infer.post_image_predict_sync",
            return_value=None,
        ):
            out = infer_tiled_sync(
                frame,
                base_url="http://example.invalid",
                predict_path="/x",
                normalize_response=lambda _d: [],
                capability="objects",
                slice_wh=64,
                overlap_wh=0,
            )
        self.assertIsNone(out)


class EmptyDetectionsTests(unittest.TestCase):
    def test_empty_vi_raw(self) -> None:
        dets = vi_raw_to_detections([], capability="objects")
        self.assertEqual(len(dets), 0)
        self.assertEqual(detections_to_vi_raw(dets, capability="objects"), [])


if __name__ == "__main__":
    unittest.main()
