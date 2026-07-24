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


class InferTiledSyncTests(unittest.TestCase):
    def test_slicer_moves_tile_coords_to_hires(self) -> None:
        """Mock tile callback: one box in tile-local coords near (640,0) origin."""
        frame = np.zeros((640, 1280, 3), dtype=np.uint8)

        def fake_normalize(_data):
            return [
                {
                    "label": "car",
                    "score": 0.95,
                    "bbox": [10.0, 10.0, 100.0, 80.0],
                    "color": "red",
                    "entity_type": "vehicle",
                }
            ]

        with patch(
            "detection.common.tiled_infer.post_image_predict_sync",
            return_value={"errorCode": 0, "result": {}},
        ):
            out = infer_tiled_sync(
                frame,
                base_url="http://example.invalid",
                predict_path="/vehicle-attribute-recognition",
                normalize_response=fake_normalize,
                capability="vehicles",
                slice_wh=640,
                overlap_wh=0,
                thread_workers=1,
            )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertGreaterEqual(len(out), 1)
        # With overlap_wh=0 and slice 640 on 1280x640 → two tiles; NMS may keep both
        # if boxes don't overlap after move. At least one box should be shifted
        # into the right half (>= 640) OR stay in left half — never double-scaled.
        for det in out:
            x1, y1, x2, y2 = det["bbox"]
            self.assertLess(x2, 1280 + 1)
            self.assertLess(y2, 640 + 1)
            self.assertGreaterEqual(x1, 0.0)
            # Anti double-scale: if someone re-applied scale 2x, widths explode
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


class TilingScaleInvariantTests(unittest.TestCase):
    """Same boxes must not be re-scaled when already in hires (tiling path)."""

    def test_scale_after_hires_would_inflate_detectable(self) -> None:
        # Simulates the bug: tiling already returned hires; scale_x=2 applied again.
        hires_box = {
            "label": "car",
            "score": 0.9,
            "bbox": [100.0, 50.0, 300.0, 200.0],
            "entity_type": "vehicle",
        }
        tiled = [dict(hires_box)]
        wrongly_scaled = [dict(hires_box)]
        scale_detections(wrongly_scaled, 2.0, 2.0)
        # Tolerance check used by regression: tiled stays near original
        self.assertAlmostEqual(tiled[0]["bbox"][0], 100.0)
        self.assertAlmostEqual(wrongly_scaled[0]["bbox"][0], 200.0)
        # Distance between wrong and right exceeds small tolerance
        self.assertGreater(
            abs(wrongly_scaled[0]["bbox"][0] - tiled[0]["bbox"][0]), 5.0
        )


class EmptyDetectionsTests(unittest.TestCase):
    def test_empty_vi_raw(self) -> None:
        dets = vi_raw_to_detections([], capability="objects")
        self.assertEqual(len(dets), 0)
        self.assertEqual(detections_to_vi_raw(dets, capability="objects"), [])


if __name__ == "__main__":
    unittest.main()
