"""Test standalone (stdlib unittest) para los helpers dual-frame de rtsp_bridge.

No forma parte de CI ni agrega dependencias nuevas (numpy ya viene con
opencv-python-headless, requirements-bridge.txt). Ejecutar manualmente:

    python test_bridge_helpers.py
"""

import os
import unittest

import numpy as np

from rtsp_bridge import (
    BRIDGE_MAX_WIDTH,
    SOURCE_URL,
    _draw_overlay,
    _extrapolate_detections,
    _is_rtsp_source,
    _is_safe_media_name,
    _maybe_resize_for_infer,
    _media_type_by_extension,
    _overlay_type_es,
    _parse_attr_labels,
    _resolve_active_source,
    _scale_detections,
)


class MaybeResizeForInferTests(unittest.TestCase):
    def test_above_threshold_downscales_and_returns_scale_factors(self) -> None:
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame_infer, scale_x, scale_y = _maybe_resize_for_infer(frame)

        self.assertEqual(frame_infer.shape[1], BRIDGE_MAX_WIDTH)
        self.assertAlmostEqual(scale_x, 1920 / BRIDGE_MAX_WIDTH)
        self.assertAlmostEqual(scale_y, 1080 / frame_infer.shape[0])

    def test_at_or_below_threshold_is_pass_through(self) -> None:
        w = BRIDGE_MAX_WIDTH
        h = max(1, round(720 * w / 1280))
        frame = np.zeros((h, w, 3), dtype=np.uint8)
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


class MediaTypeByExtensionTests(unittest.TestCase):
    """Helper puro (sin cv2) usado por el resolver de fuente (overlay-preview)."""

    def test_video_extensions(self) -> None:
        for name in ("clip.mp4", "clip.AVI", "clip.mov", "clip.mkv"):
            self.assertEqual(_media_type_by_extension(name), "video")

    def test_image_extensions(self) -> None:
        for name in ("photo.jpg", "photo.JPEG", "photo.png", "photo.bmp"):
            self.assertEqual(_media_type_by_extension(name), "image")

    def test_unknown_extension_is_none(self) -> None:
        self.assertIsNone(_media_type_by_extension("readme.txt"))


class SafeMediaNameTests(unittest.TestCase):
    """Allow-list guard local (defensa en profundidad, LMP-1)."""

    def test_plain_basename_is_safe(self) -> None:
        self.assertTrue(_is_safe_media_name("Brasil6.mp4"))

    def test_path_traversal_rejected(self) -> None:
        self.assertFalse(_is_safe_media_name("../etc/passwd"))
        self.assertFalse(_is_safe_media_name("sub/dir/file.mp4"))
        self.assertFalse(_is_safe_media_name(".."))
        self.assertFalse(_is_safe_media_name(""))


class ResolveActiveSourceTests(unittest.TestCase):
    """Selector RTSP vs. archivo local (BCS-1, BCS-2)."""

    def test_rtsp_url_detected(self) -> None:
        self.assertTrue(_is_rtsp_source("rtsp://mediamtx:8554/webcam"))
        self.assertFalse(_is_rtsp_source("/media/videos/Brasil6.mp4"))

    def test_selected_video_resolves_local_non_rtsp(self) -> None:
        source, is_rtsp, is_image = _resolve_active_source(
            {"name": "Brasil6.mp4", "type": "video"}
        )

        self.assertFalse(is_rtsp)
        self.assertFalse(is_image)
        self.assertTrue(source.endswith(os.path.join("videos", "Brasil6.mp4")))

    def test_selected_photo_resolves_image_branch(self) -> None:
        source, is_rtsp, is_image = _resolve_active_source(
            {"name": "sample.jpg", "type": "image"}
        )

        self.assertFalse(is_rtsp)
        self.assertTrue(is_image)
        self.assertTrue(source.endswith(os.path.join("images", "sample.jpg")))

    def test_no_selection_falls_back_to_source_url(self) -> None:
        source, is_rtsp, is_image = _resolve_active_source(None)

        self.assertEqual(source, SOURCE_URL)
        self.assertFalse(is_image)


class DrawOverlayTests(unittest.TestCase):
    """Bbox+label sobre copia del frame (FO-1); no-op sin detecciones."""

    def test_no_detections_returns_same_frame_unmodified(self) -> None:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = _draw_overlay(frame, [])

        self.assertIs(result, frame)

    def test_with_detections_returns_modified_copy(self) -> None:
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        dets = [
            {
                "track_id": "1",
                "label": "car",
                "score": 0.9,
                "bbox": [10.0, 10.0, 100.0, 100.0],
                "plate": {"text": "ABC123", "score": 0.8},
            }
        ]

        result = _draw_overlay(frame, dets)

        self.assertIsNot(result, frame)
        self.assertTrue(np.any(result != frame))


class OverlaySpanishLabelsTests(unittest.TestCase):
    def test_overlay_maps_vehicle_to_spanish(self) -> None:
        self.assertEqual(_overlay_type_es("vehicle"), "vehiculo")
        self.assertEqual(_overlay_type_es("car"), "auto")

    def test_overlay_strips_chinese_parenthetical(self) -> None:
        self.assertEqual(_overlay_type_es("sedan(轿车)"), "sedan")

    def test_parse_attr_ignores_pure_chinese(self) -> None:
        color, vtype = _parse_attr_labels(["红色", "轿车"], [0.9, 0.8])
        self.assertIsNone(color)
        self.assertIsNone(vtype)

    def test_parse_attr_keeps_english_prefix(self) -> None:
        color, vtype = _parse_attr_labels(["red(红色)", "sedan(轿车)"], [0.9, 0.8])
        self.assertEqual(color, "red")
        self.assertEqual(vtype, "sedan")


class ExtrapolateDetectionsTests(unittest.TestCase):
    def test_projects_bbox_forward_with_constant_velocity(self) -> None:
        prev = [{"track_id": "1", "bbox": [0.0, 0.0, 10.0, 10.0], "label": "car"}]
        curr = [{"track_id": "1", "bbox": [10.0, 0.0, 20.0, 10.0], "label": "car"}]
        # Velocidad +10 px/s en x; 0.5s adelante => +5
        out = _extrapolate_detections(curr, 1.0, prev, 0.0, 1.5, max_ahead=1.0)
        self.assertEqual(out[0]["bbox"], [15.0, 0.0, 25.0, 10.0])

    def test_no_prev_returns_curr_unchanged(self) -> None:
        curr = [{"track_id": "1", "bbox": [10.0, 0.0, 20.0, 10.0]}]
        out = _extrapolate_detections(curr, 1.0, None, 0.0, 1.5)
        self.assertEqual(out[0]["bbox"], [10.0, 0.0, 20.0, 10.0])

    def test_stale_returns_empty(self) -> None:
        curr = [{"track_id": "1", "bbox": [10.0, 0.0, 20.0, 10.0]}]
        out = _extrapolate_detections(
            curr, 1.0, None, 0.0, 5.0, max_ahead=2.5, stale_max=3.0
        )
        self.assertEqual(out, [])

    def test_projects_up_to_higher_max_ahead(self) -> None:
        prev = [{"track_id": "1", "bbox": [0.0, 0.0, 10.0, 10.0], "label": "car"}]
        curr = [{"track_id": "1", "bbox": [10.0, 0.0, 20.0, 10.0], "label": "car"}]
        # 2.0s adelante a 10 px/s => +20 (antes el tope 0.6 dejaba solo +6)
        out = _extrapolate_detections(
            curr, 1.0, prev, 0.0, 3.0, max_ahead=2.5, stale_max=3.0
        )
        self.assertEqual(out[0]["bbox"], [30.0, 0.0, 40.0, 10.0])


if __name__ == "__main__":
    unittest.main()
