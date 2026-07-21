"""Test standalone (stdlib unittest) para helpers del bridge (solo fotos + nativo).

No forma parte de CI ni agrega dependencias nuevas (numpy ya viene con
opencv-python-headless, requirements-bridge.txt). Ejecutar manualmente:

    python test_bridge_helpers.py
"""

import base64
import os
import unittest

import numpy as np

from rtsp_bridge import (
    BRIDGE_MAX_WIDTH,
    RTSP_URL,
    SOURCE_URL,
    _decode_paddlex_result_image,
    _draw_preview,
    _is_rtsp_source,
    _is_safe_media_name,
    _maybe_resize_for_infer,
    _media_type_by_extension,
    _normalize_paddlex_result,
    _parse_attr_labels,
    _preview_box_color,
    _preview_label,
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
        self.assertEqual(dets[0]["track_id"], "1")
        self.assertEqual(dets[0]["label"], "car")
        self.assertIsNone(dets[0]["plate"])

    def test_pass_through_when_no_resize(self) -> None:
        dets = [{"bbox": [1.0, 2.0, 3.0, 4.0], "track_id": "1"}]
        result = _scale_detections(dets, 1.0, 1.0)

        self.assertEqual(result[0]["bbox"], [1.0, 2.0, 3.0, 4.0])


class MediaTypeByExtensionTests(unittest.TestCase):
    """Solo imágenes en el selector; video local ya no es un tipo de muestra."""

    def test_image_extensions(self) -> None:
        for name in ("photo.jpg", "photo.JPEG", "photo.png", "photo.bmp"):
            self.assertEqual(_media_type_by_extension(name), "image")

    def test_video_extension_is_not_image(self) -> None:
        self.assertIsNone(_media_type_by_extension("clip.mp4"))

    def test_unknown_extension_is_none(self) -> None:
        self.assertIsNone(_media_type_by_extension("readme.txt"))


class SafeMediaNameTests(unittest.TestCase):
    def test_plain_basename_is_safe(self) -> None:
        self.assertTrue(_is_safe_media_name("sample_mvi20011_img00001.jpg"))

    def test_path_traversal_rejected(self) -> None:
        self.assertFalse(_is_safe_media_name("../etc/passwd"))
        self.assertFalse(_is_safe_media_name("sub/dir/file.jpg"))
        self.assertFalse(_is_safe_media_name(".."))
        self.assertFalse(_is_safe_media_name(""))


class ResolveActiveSourceTests(unittest.TestCase):
    def test_rtsp_url_detected(self) -> None:
        self.assertTrue(_is_rtsp_source("rtsp://mediamtx:8554/webcam"))
        self.assertFalse(_is_rtsp_source("/media/images/sample.jpg"))

    def test_selected_photo_resolves_image_branch(self) -> None:
        source, is_rtsp, is_image = _resolve_active_source(
            {"name": "sample.jpg", "type": "image"}
        )

        self.assertFalse(is_rtsp)
        self.assertTrue(is_image)
        self.assertTrue(source.endswith(os.path.join("images", "sample.jpg")))

    def test_selected_video_is_ignored_falls_back(self) -> None:
        source, is_rtsp, is_image = _resolve_active_source(
            {"name": "Brasil6.mp4", "type": "video"}
        )

        self.assertEqual(source, SOURCE_URL if _is_rtsp_source(SOURCE_URL) else RTSP_URL)
        self.assertTrue(is_rtsp)
        self.assertFalse(is_image)

    def test_no_selection_falls_back_to_source_url(self) -> None:
        source, is_rtsp, is_image = _resolve_active_source(None)

        if _is_rtsp_source(SOURCE_URL):
            self.assertEqual(source, SOURCE_URL)
            self.assertTrue(is_rtsp)
            self.assertFalse(is_image)
        elif _media_type_by_extension(SOURCE_URL) == "image":
            self.assertEqual(source, SOURCE_URL)
            self.assertTrue(is_image)
        else:
            self.assertEqual(source, RTSP_URL)
            self.assertTrue(is_rtsp)


class DecodePaddlexResultImageTests(unittest.TestCase):
    def test_decodes_plain_base64(self) -> None:
        payload = b"\xff\xd8\xfffake-jpeg"
        data = {"result": {"vehicles": [], "image": base64.b64encode(payload).decode()}}
        self.assertEqual(_decode_paddlex_result_image(data), payload)

    def test_decodes_data_uri(self) -> None:
        payload = b"jpeg-bytes"
        b64 = base64.b64encode(payload).decode()
        data = {"result": {"image": f"data:image/jpeg;base64,{b64}"}}
        self.assertEqual(_decode_paddlex_result_image(data), payload)

    def test_missing_image_returns_none(self) -> None:
        self.assertIsNone(_decode_paddlex_result_image({"result": {"vehicles": []}}))
        self.assertIsNone(_decode_paddlex_result_image({"result": {"image": ""}}))


class NormalizePaddlexResultTests(unittest.TestCase):
    def test_vehicles_shape_yields_detections(self) -> None:
        data = {
            "result": {
                "vehicles": [
                    {
                        "bbox": [10, 20, 30, 40],
                        "score": 0.91,
                        "attributes": [
                            {"label": "red(红色)", "score": 0.9},
                            {"label": "sedan(轿车)", "score": 0.8},
                        ],
                    }
                ],
                "image": base64.b64encode(b"x").decode(),
            }
        }
        dets = _normalize_paddlex_result(data)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["label"], "sedan")
        self.assertEqual(dets[0]["color"], "red")
        self.assertEqual(dets[0]["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertIn("track_id", dets[0])


class AttrLabelTests(unittest.TestCase):
    def test_parse_attr_ignores_pure_chinese(self) -> None:
        color, vtype = _parse_attr_labels(["红色", "轿车"], [0.9, 0.8])
        self.assertIsNone(color)
        self.assertIsNone(vtype)

    def test_parse_attr_keeps_english_prefix(self) -> None:
        color, vtype = _parse_attr_labels(["red(红色)", "sedan(轿车)"], [0.9, 0.8])
        self.assertEqual(color, "red")
        self.assertEqual(vtype, "sedan")


class PreviewLabelTests(unittest.TestCase):
    def test_english_parts_only(self) -> None:
        self.assertEqual(
            _preview_label({"label": "suv", "color": "black"}),
            "suv black",
        )

    def test_fallback_vehicle(self) -> None:
        self.assertEqual(_preview_label({}), "vehicle")

    def test_no_cjk_in_label(self) -> None:
        text = _preview_label({"label": "sedan", "color": "golden"})
        self.assertNotRegex(text, r"[\u4e00-\u9fff]")


class PreviewBoxColorTests(unittest.TestCase):
    def test_types_get_distinct_colors(self) -> None:
        sedan = _preview_box_color({"label": "sedan"})
        suv = _preview_box_color({"label": "suv"})
        truck = _preview_box_color({"label": "truck"})
        self.assertNotEqual(sedan, suv)
        self.assertNotEqual(suv, truck)
        self.assertNotEqual(sedan, (0, 220, 0))

    def test_unknown_uses_track_palette(self) -> None:
        a = _preview_box_color({"label": "weird", "track_id": "1"})
        b = _preview_box_color({"label": "weird", "track_id": "2"})
        self.assertNotEqual(a, b)


class DrawPreviewTests(unittest.TestCase):
    def test_returns_jpeg_bytes(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        dets = [
            {
                "label": "suv",
                "color": "black",
                "bbox": [10.0, 20.0, 80.0, 90.0],
                "score": 0.9,
                "track_id": "1",
            }
        ]
        jpeg = _draw_preview(frame, dets)
        self.assertIsNotNone(jpeg)
        assert jpeg is not None
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
