"""Test standalone (stdlib unittest) para los helpers dual-frame de rtsp_bridge.

No forma parte de CI ni agrega dependencias nuevas (numpy ya viene con
opencv-python-headless, requirements-bridge.txt). Ejecutar manualmente:

    python test_bridge_helpers.py
"""

import unittest

import numpy as np

from rtsp_bridge import BRIDGE_MAX_WIDTH, _maybe_resize_for_infer, _scale_detections


class MaybeResizeForInferTests(unittest.TestCase):
    def test_above_threshold_downscales_and_returns_scale_factors(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_infer, scale_x, scale_y = _maybe_resize_for_infer(frame)

        self.assertEqual(frame_infer.shape[1], BRIDGE_MAX_WIDTH)
        self.assertAlmostEqual(scale_x, 1920 / BRIDGE_MAX_WIDTH)
        self.assertAlmostEqual(scale_y, 1080 / frame_infer.shape[0])

    def test_at_or_below_threshold_is_pass_through(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame_infer, scale_x, scale_y = _maybe_resize_for_infer(frame)

        self.assertIs(frame_infer, frame)
        self.assertEqual((scale_x, scale_y), (1.0, 1.0))


class ScaleDetectionsTests(unittest.TestCase):
    def test_round_trip_scales_bbox_back_to_hires(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, scale_x, scale_y = _maybe_resize_for_infer(frame)
        dets = [
            {
                "track_id": "1",
                "label": "car",
                "score": 0.9,
                "color": "white",
                "bbox": [100.0, 50.0, 300.0, 200.0],
                "plate": None,
                "frame_ts": "now",
            }
        ]

        _scale_detections(dets, scale_x, scale_y)

        x1, y1, x2, y2 = dets[0]["bbox"]
        self.assertAlmostEqual(x1, 100.0 * scale_x)
        self.assertAlmostEqual(y1, 50.0 * scale_y)
        self.assertAlmostEqual(x2, 300.0 * scale_x)
        self.assertAlmostEqual(y2, 200.0 * scale_y)
        # No debe tocar ninguna otra clave del dict de detección.
        self.assertEqual(dets[0]["track_id"], "1")
        self.assertEqual(dets[0]["label"], "car")
        self.assertIsNone(dets[0]["plate"])

    def test_pass_through_when_no_resize(self) -> None:
        dets = [{"bbox": [1.0, 2.0, 3.0, 4.0], "track_id": "1"}]
        result = _scale_detections(dets, 1.0, 1.0)

        self.assertEqual(result[0]["bbox"], [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    unittest.main()
