"""Test standalone para helpers de detection/* y bridge.media (foto-only).

Ejecutar desde la raíz del repo (PYTHONPATH=.):

    PYTHONPATH=. python3 tests/test_bridge_helpers.py
"""

from __future__ import annotations

import base64
import os
import unittest

import numpy as np

from bridge.media import (
    is_safe_media_name,
    media_type_by_extension,
    resolve_active_source,
)
from detection.common.geometry import (
    BRIDGE_MAX_WIDTH,
    maybe_resize_for_infer,
    scale_detections,
)
from detection.common.preview import draw_preview, preview_box_color, preview_label
from detection.faces import normalize_face_result
from detection.objects import merge_coco_detections, normalize_object_detection_result
from detection.pedestrians import merge_person_attributes, parse_person_attributes
from detection.scene import (
    class_ratios_from_label_map,
    infer_scene_type,
    normalize_scene_result,
)
from detection.vehicles import (
    decode_paddlex_result_image,
    normalize_vehicle_result,
    parse_attr_labels,
)


class MaybeResizeForInferTests(unittest.TestCase):
    def test_above_threshold_downscales_and_returns_scale_factors(self) -> None:
        # Wider than default BRIDGE_MAX_WIDTH (1920) so downscale is exercised.
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        frame_infer, scale_x, scale_y = maybe_resize_for_infer(frame)

        self.assertEqual(frame_infer.shape[1], BRIDGE_MAX_WIDTH)
        self.assertAlmostEqual(scale_x, 3840 / BRIDGE_MAX_WIDTH)
        self.assertAlmostEqual(scale_y, 2160 / frame_infer.shape[0])

    def test_at_or_below_threshold_is_pass_through(self) -> None:
        w = BRIDGE_MAX_WIDTH
        h = max(1, round(720 * w / 1280))
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame_infer, scale_x, scale_y = maybe_resize_for_infer(frame)

        self.assertIs(frame_infer, frame)
        self.assertEqual((scale_x, scale_y), (1.0, 1.0))


class ScaleDetectionsTests(unittest.TestCase):
    def test_round_trip_scales_bbox_back_to_hires(self) -> None:
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        _, scale_x, scale_y = maybe_resize_for_infer(frame)
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

        scale_detections(dets, scale_x, scale_y)

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
        result = scale_detections(dets, 1.0, 1.0)

        self.assertEqual(result[0]["bbox"], [1.0, 2.0, 3.0, 4.0])


class MediaTypeByExtensionTests(unittest.TestCase):
    def test_image_extensions(self) -> None:
        for name in ("photo.jpg", "photo.JPEG", "photo.png", "photo.bmp"):
            self.assertEqual(media_type_by_extension(name), "image")

    def test_video_extension_is_not_image(self) -> None:
        self.assertIsNone(media_type_by_extension("clip.mp4"))

    def test_unknown_extension_is_none(self) -> None:
        self.assertIsNone(media_type_by_extension("readme.txt"))


class SafeMediaNameTests(unittest.TestCase):
    def test_plain_basename_is_safe(self) -> None:
        self.assertTrue(is_safe_media_name("sample_mvi20011_img00001.jpg"))

    def test_path_traversal_rejected(self) -> None:
        self.assertFalse(is_safe_media_name("../etc/passwd"))
        self.assertFalse(is_safe_media_name("sub/dir/file.jpg"))
        self.assertFalse(is_safe_media_name(".."))
        self.assertFalse(is_safe_media_name(""))


class ResolveActiveSourceTests(unittest.TestCase):
    def test_selected_photo_resolves_image_path(self) -> None:
        source = resolve_active_source({"name": "sample.jpg", "type": "image"})

        self.assertIsNotNone(source)
        self.assertTrue(source.endswith(os.path.join("images", "sample.jpg")))

    def test_selected_video_is_ignored_idle(self) -> None:
        source = resolve_active_source({"name": "Brasil6.mp4", "type": "video"})

        self.assertIsNone(source)

    def test_no_selection_is_idle(self) -> None:
        self.assertIsNone(resolve_active_source(None))
        self.assertIsNone(resolve_active_source({"name": None}))
        self.assertIsNone(resolve_active_source({}))


class DecodePaddlexResultImageTests(unittest.TestCase):
    def test_decodes_plain_base64(self) -> None:
        payload = b"\xff\xd8\xfffake-jpeg"
        data = {"result": {"vehicles": [], "image": base64.b64encode(payload).decode()}}
        self.assertEqual(decode_paddlex_result_image(data), payload)

    def test_decodes_data_uri(self) -> None:
        payload = b"jpeg-bytes"
        b64 = base64.b64encode(payload).decode()
        data = {"result": {"image": f"data:image/jpeg;base64,{b64}"}}
        self.assertEqual(decode_paddlex_result_image(data), payload)

    def test_missing_image_returns_none(self) -> None:
        self.assertIsNone(decode_paddlex_result_image({"result": {"vehicles": []}}))
        self.assertIsNone(decode_paddlex_result_image({"result": {"image": ""}}))


class NormalizeVehicleResultTests(unittest.TestCase):
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
        dets = normalize_vehicle_result(data)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["label"], "sedan")
        self.assertEqual(dets[0]["color"], "red")
        self.assertEqual(dets[0]["bbox"], [10.0, 20.0, 30.0, 40.0])
        self.assertIn("track_id", dets[0])


class AttrLabelTests(unittest.TestCase):
    def test_parse_attr_ignores_pure_chinese(self) -> None:
        color, vtype = parse_attr_labels(["红色", "轿车"], [0.9, 0.8])
        self.assertIsNone(color)
        self.assertIsNone(vtype)

    def test_parse_attr_keeps_english_prefix(self) -> None:
        color, vtype = parse_attr_labels(["red(红色)", "sedan(轿车)"], [0.9, 0.8])
        self.assertEqual(color, "red")
        self.assertEqual(vtype, "sedan")


class PreviewLabelTests(unittest.TestCase):
    def test_english_parts_only(self) -> None:
        self.assertEqual(
            preview_label({"label": "suv", "color": "black"}),
            "suv black",
        )

    def test_fallback_vehicle(self) -> None:
        self.assertEqual(preview_label({}), "vehicle")

    def test_no_cjk_in_label(self) -> None:
        text = preview_label({"label": "sedan", "color": "golden"})
        self.assertNotRegex(text, r"[\u4e00-\u9fff]")


class PreviewBoxColorTests(unittest.TestCase):
    def test_types_get_distinct_colors(self) -> None:
        sedan = preview_box_color({"label": "sedan"})
        suv = preview_box_color({"label": "suv"})
        truck = preview_box_color({"label": "truck"})
        self.assertNotEqual(sedan, suv)
        self.assertNotEqual(suv, truck)
        self.assertNotEqual(sedan, (0, 220, 0))

    def test_unknown_uses_track_palette(self) -> None:
        a = preview_box_color({"label": "weird", "track_id": "1"})
        b = preview_box_color({"label": "weird", "track_id": "2"})
        self.assertNotEqual(a, b)


class NormalizeObjectDetectionResultTests(unittest.TestCase):
    def test_single_label_box_parses_correctly(self) -> None:
        data = {
            "result": {
                "boxes": [
                    {
                        "cls_id": 0,
                        "label": "person",
                        "score": 0.87,
                        "coordinate": [10, 20, 50, 90],
                    }
                ]
            }
        }
        dets = normalize_object_detection_result(data)

        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["label"], "person")
        self.assertAlmostEqual(dets[0]["score"], 0.87)
        self.assertEqual(dets[0]["bbox"], [10.0, 20.0, 50.0, 90.0])
        self.assertEqual(dets[0]["entity_type"], "object")
        self.assertNotIn("color", dets[0])
        self.assertNotIn("plate", dets[0])

    def test_missing_boxes_returns_empty(self) -> None:
        self.assertEqual(normalize_object_detection_result({"result": {}}), [])

    def test_paddlex_detected_objects_category_name(self) -> None:
        """PaddleX object_detection returns detectedObjects + categoryName/bbox."""
        data = {
            "result": {
                "detectedObjects": [
                    {
                        "bbox": [133.6, 12.5, 447.0, 445.1],
                        "categoryId": 0,
                        "categoryName": "person",
                        "score": 0.895,
                    }
                ],
                "image": "unused",
            }
        }
        dets = normalize_object_detection_result(data)

        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["label"], "person")
        self.assertAlmostEqual(dets[0]["score"], 0.895)
        self.assertEqual(dets[0]["bbox"], [133.6, 12.5, 447.0, 445.1])
        self.assertEqual(dets[0]["entity_type"], "object")


class MergeCocoDetectionsTests(unittest.TestCase):
    @staticmethod
    def _vehicle_det(bbox: list[float]) -> dict:
        return {
            "track_id": "v-1",
            "label": "sedan",
            "score": 0.9,
            "color": "white",
            "bbox": bbox,
            "plate": None,
            "frame_ts": "now",
            "entity_type": "vehicle",
        }

    @staticmethod
    def _object_det(label: str, bbox: list[float], track_id: str = "o-1") -> dict:
        return {
            "track_id": track_id,
            "label": label,
            "score": 0.8,
            "bbox": bbox,
            "entity_type": "object",
            "frame_ts": "now",
        }

    def test_dedupes_vehicle_class_high_iou(self) -> None:
        vehicle = [self._vehicle_det([10.0, 10.0, 100.0, 100.0])]
        obj = [self._object_det("car", [12.0, 12.0, 98.0, 98.0])]

        merged = merge_coco_detections(vehicle, obj)

        self.assertEqual(merged, [])

    def test_keeps_non_vehicle_class_even_with_high_iou(self) -> None:
        vehicle = [self._vehicle_det([10.0, 10.0, 100.0, 100.0])]
        obj = [self._object_det("person", [12.0, 12.0, 98.0, 98.0])]

        merged = merge_coco_detections(vehicle, obj)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["label"], "person")

    def test_keeps_vehicle_class_with_low_iou(self) -> None:
        vehicle = [self._vehicle_det([10.0, 10.0, 100.0, 100.0])]
        obj = [self._object_det("car", [500.0, 500.0, 600.0, 600.0])]

        merged = merge_coco_detections(vehicle, obj)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["label"], "car")

    def test_empty_vehicle_list_keeps_all_object_detections(self) -> None:
        obj = [
            self._object_det("car", [1.0, 1.0, 5.0, 5.0]),
            self._object_det("person", [10.0, 10.0, 20.0, 20.0], track_id="o-2"),
        ]

        merged = merge_coco_detections([], obj)

        self.assertEqual(len(merged), 2)


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
        jpeg = draw_preview(frame, dets)
        self.assertIsNotNone(jpeg)
        assert jpeg is not None
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))

    def test_scene_badge_still_returns_jpeg(self) -> None:
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        dets = [
            {
                "track_id": "scene-0",
                "entity_type": "scene",
                "label": "street",
                "score": 0.7,
                "bbox": [0, 0, 100, 80],
                "scene": {"type": "street"},
            }
        ]
        jpeg = draw_preview(frame, dets)
        self.assertIsNotNone(jpeg)


class NormalizeFaceResultTests(unittest.TestCase):
    def test_parses_boxes(self) -> None:
        data = {
            "result": {
                "boxes": [
                    {"score": 0.95, "coordinate": [1, 2, 30, 40]},
                ]
            }
        }
        dets = normalize_face_result(data)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["entity_type"], "face")
        self.assertEqual(dets[0]["label"], "face")
        self.assertTrue(str(dets[0]["track_id"]).startswith("f-"))


class MergePersonAttributesTests(unittest.TestCase):
    def test_enriches_matching_person(self) -> None:
        objects = [
            {
                "track_id": "o-1",
                "label": "person",
                "score": 0.9,
                "bbox": [10.0, 10.0, 50.0, 100.0],
                "entity_type": "object",
            }
        ]
        attrs = [
            {
                "label": "person",
                "score": 0.8,
                "bbox": [12.0, 12.0, 48.0, 98.0],
                "person": {"gender": "female", "age_group": "adult"},
                "entity_type": "object",
            }
        ]
        merged = merge_person_attributes(objects, attrs)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["person"]["gender"], "female")

    def test_orphan_attr_appended(self) -> None:
        objects = [
            {
                "track_id": "o-1",
                "label": "dog",
                "score": 0.9,
                "bbox": [10.0, 10.0, 20.0, 20.0],
                "entity_type": "object",
            }
        ]
        attrs = [
            {
                "label": "person",
                "score": 0.7,
                "bbox": [100.0, 100.0, 140.0, 200.0],
                "person": {"gender": "male"},
                "entity_type": "object",
            }
        ]
        merged = merge_person_attributes(objects, attrs)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[1]["label"], "person")
        self.assertEqual(merged[1]["person"]["gender"], "male")


class ParsePersonAttributesTests(unittest.TestCase):
    def test_gender_and_age(self) -> None:
        attrs = parse_person_attributes(
            ["Female(女)", "Adult(成人)", "front"],
            [0.9, 0.8, 0.7],
        )
        self.assertEqual(attrs["gender"], "female")
        self.assertEqual(attrs["age_group"], "adult")
        self.assertEqual(attrs["direction"], "front")


class SceneHeuristicsTests(unittest.TestCase):
    def test_highway_from_ratios(self) -> None:
        stype, conf = infer_scene_type(
            {"road": 0.45, "sidewalk": 0.01, "building": 0.05, "sky": 0.2}
        )
        self.assertEqual(stype, "highway")
        self.assertGreaterEqual(conf, 0.4)

    def test_street_from_ratios(self) -> None:
        stype, _ = infer_scene_type(
            {"road": 0.25, "sidewalk": 0.08, "building": 0.3, "sky": 0.1}
        )
        self.assertEqual(stype, "street")

    def test_normalize_scene_cityscapes(self) -> None:
        # 100 px: 60 road, 20 sidewalk, 20 building
        label_map = [0] * 60 + [1] * 20 + [2] * 20
        data = {"result": {"labelMap": label_map, "shape": [10, 10]}}
        det = normalize_scene_result(data, frame_wh=(10, 10), label_mode="cityscapes")
        self.assertIsNotNone(det)
        assert det is not None
        self.assertEqual(det["entity_type"], "scene")
        self.assertEqual(det["track_id"], "scene-0")
        self.assertIn(det["label"], {"street", "highway", "parking", "rural", "unknown"})
        self.assertIsNone(det["scene"]["crosswalk"])
        self.assertIsNone(det["scene"]["lanes"])

    def test_normalize_scene_lane_mode(self) -> None:
        label_map = [0] * 80 + [2] * 15 + [3] * 5
        data = {"result": {"labelMap": label_map}}
        det = normalize_scene_result(data, frame_wh=(20, 5), label_mode="lane")
        self.assertIsNotNone(det)
        assert det is not None
        self.assertIsNotNone(det["scene"]["lanes"])
        self.assertTrue(det["scene"]["lanes"]["present"])


    def test_bdd_marks_crosswalk(self) -> None:
        label_map = [0] * 40 + [6] * 30 + [8] * 30
        data = {"result": {"labelMap": label_map}}
        det = normalize_scene_result(
            data, frame_wh=(10, 10), label_mode="bdd_marks"
        )
        self.assertIsNotNone(det)
        assert det is not None
        self.assertIsNotNone(det["scene"]["crosswalk"])
        self.assertTrue(det["scene"]["crosswalk"]["present"])

    def test_class_ratios(self) -> None:
        ratios = class_ratios_from_label_map(
            [0, 0, 1, 1], {0: "road", 1: "sidewalk"}
        )
        self.assertAlmostEqual(ratios["road"], 0.5)
        self.assertAlmostEqual(ratios["sidewalk"], 0.5)


class NormalizePoseAndTextTests(unittest.TestCase):
    def test_pose_boxes(self) -> None:
        from detection.pose import normalize_pose_result

        dets = normalize_pose_result(
            {
                "result": {
                    "boxes": [
                        {
                            "score": 0.9,
                            "coordinate": [1, 2, 30, 80],
                            "keypoints": [[1, 2], [30, 80]],
                        }
                    ]
                }
            }
        )
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["entity_type"], "pose")
        self.assertTrue(str(dets[0]["track_id"]).startswith("k-"))

    def test_scene_ocr_lines(self) -> None:
        from detection.text import normalize_scene_ocr_result

        dets = normalize_scene_ocr_result(
            {
                "result": {
                    "ocrResults": [
                        {
                            "prunedResult": {
                                "rec_texts": ["STOP", "ABC"],
                                "rec_scores": [0.95, 0.4],
                                "dt_polys": [
                                    [[0, 0], [10, 0], [10, 5], [0, 5]],
                                    [[0, 0], [1, 0], [1, 1], [0, 1]],
                                ],
                            }
                        }
                    ]
                }
            }
        )
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["text"], "STOP")
        self.assertEqual(dets[0]["entity_type"], "text")

    def test_signs_filter(self) -> None:
        from detection.signs import normalize_signs_result

        dets = normalize_signs_result(
            {
                "result": {
                    "boxes": [
                        {
                            "label": "stop sign",
                            "score": 0.9,
                            "coordinate": [1, 2, 3, 4],
                        },
                        {
                            "label": "person",
                            "score": 0.9,
                            "coordinate": [5, 6, 7, 8],
                        },
                    ]
                }
            }
        )
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["entity_type"], "sign")

if __name__ == "__main__":
    unittest.main()
