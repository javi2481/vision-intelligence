"""Tests PR3: NMS-B cross-cap + PolygonZone extent.

Ejecutar: PYTHONPATH=. python3 tests/test_nms_zones_pr3.py
"""

from __future__ import annotations

import unittest

import numpy as np
import supervision as sv

from adapter.epp_core import PerceptionEvent, SCHEMA_VERSION, _normalize_detection
from detection.common.nms_cross_cap import (
    apply_cross_cap_nms,
    class_id_for_cross_cap_nms,
    normalize_detections_data_keys,
    reset_cross_cap_class_ids,
    vi_det_to_detections,
)
from detection.common.tiled_infer import class_id_for_tile_nms
from detection.common.zones import (
    ZoneConfig,
    denormalize_polygon,
    tag_detections_with_zones,
)
from rules.app import PerceptionEventIn, evaluate_rule


class CrossCapClassIdTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_cross_cap_class_ids()

    def test_remap_uses_entity_type_not_tile_label_map(self) -> None:
        # Tile map numeraría "person"/"face" por capability/label distinto.
        tile_person = class_id_for_tile_nms("person", capability="objects")
        tile_face = class_id_for_tile_nms("face", capability="faces")
        # Cross-cap: entity_type object vs face — ids propios del mapa B.
        cross_obj = class_id_for_cross_cap_nms("object")
        cross_face = class_id_for_cross_cap_nms("face")
        self.assertNotEqual(cross_obj, cross_face)
        # El id de tile no es la fuente de verdad de B (pueden coincidir
        # numéricamente por azar; lo crítico es que B discrimine por entity).
        self.assertEqual(class_id_for_cross_cap_nms("object"), cross_obj)
        self.assertEqual(class_id_for_cross_cap_nms("face"), cross_face)
        # Remap: vi_det_to_detections debe usar cross-cap, no tile.
        face = vi_det_to_detections(
            {
                "track_id": "f-1",
                "label": "face",
                "score": 0.9,
                "bbox": [10, 10, 40, 40],
                "entity_type": "face",
            }
        )
        self.assertEqual(int(face.class_id[0]), cross_face)
        self.assertNotEqual(int(face.class_id[0]), tile_person)
        _ = tile_face  # usado arriba para poblar mapa A


class NmsCrossCapBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_cross_cap_class_ids()

    def test_face_and_person_survive_overlap(self) -> None:
        dets = [
            {
                "track_id": "f-1",
                "label": "face",
                "score": 0.95,
                "bbox": [100.0, 100.0, 200.0, 200.0],
                "entity_type": "face",
            },
            {
                "track_id": "o-1",
                "label": "person",
                "score": 0.90,
                "bbox": [105.0, 105.0, 205.0, 205.0],
                "entity_type": "object",
            },
        ]
        kept = apply_cross_cap_nms(dets, threshold=0.5)
        types = {d["entity_type"] for d in kept}
        self.assertEqual(types, {"face", "object"})
        self.assertEqual(len(kept), 2)

    def test_duplicate_objects_deduped_survivor_keeps_track_and_plate(self) -> None:
        dets = [
            {
                "track_id": "o-low",
                "label": "bottle",
                "score": 0.60,
                "bbox": [50.0, 50.0, 150.0, 150.0],
                "entity_type": "object",
                "plate": {"text": "IGNORE", "score": 0.1},
            },
            {
                "track_id": "o-hi",
                "label": "bottle",
                "score": 0.92,
                "bbox": [52.0, 52.0, 152.0, 152.0],
                "entity_type": "object",
                "plate": {"text": "KEEP99", "score": 0.8},
            },
        ]
        kept = apply_cross_cap_nms(dets, threshold=0.5)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["track_id"], "o-hi")
        self.assertEqual(kept[0]["plate"]["text"], "KEEP99")
        self.assertAlmostEqual(float(kept[0]["score"]), 0.92)

    def test_append_one_without_bbox_excluded_from_nms_but_kept(self) -> None:
        dets = [
            {
                "track_id": "s-1",
                "label": "street",
                "score": 0.8,
                "entity_type": "scene",
                "scene": {"type": "street"},
            },
            {
                "track_id": "v-1",
                "label": "car",
                "score": 0.9,
                "bbox": [10.0, 10.0, 80.0, 80.0],
                "entity_type": "vehicle",
            },
        ]
        kept = apply_cross_cap_nms(dets)
        self.assertEqual(len(kept), 2)
        scene = next(d for d in kept if d["entity_type"] == "scene")
        self.assertNotIn("bbox", scene)

    def test_heterogeneous_data_keys_normalize_then_merge(self) -> None:
        a = sv.Detections(
            xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([0], dtype=np.int32),
            data={"track_id": np.array(["a"], dtype=object)},
        )
        b = sv.Detections(
            xyxy=np.array([[1, 1, 11, 11]], dtype=np.float32),
            confidence=np.array([0.8], dtype=np.float32),
            class_id=np.array([0], dtype=np.int32),
            data={"plate": np.array([{"text": "X"}], dtype=object)},
        )
        with self.assertRaises(ValueError):
            sv.Detections.merge([a, b])
        normed = normalize_detections_data_keys([a, b])
        merged = sv.Detections.merge(normed)
        self.assertEqual(len(merged), 2)
        self.assertIn("track_id", merged.data)
        self.assertIn("plate", merged.data)


class ZoneExtentTests(unittest.TestCase):
    def test_hit_and_miss_two_frame_sizes(self) -> None:
        # Cuadrante superior-izquierdo normalizado.
        zone = ZoneConfig(
            id="no_parking",
            polygon_norm=np.array(
                [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]], dtype=np.float64
            ),
        )
        for wh in ((200, 200), (800, 600)):
            w, h = wh
            # BOTTOM_CENTER dentro del polígono.
            inside = {
                "track_id": "v-in",
                "score": 0.9,
                "bbox": [0.05 * w, 0.05 * h, 0.15 * w, 0.2 * h],
                "entity_type": "vehicle",
            }
            # BOTTOM_CENTER fuera (abajo-derecha).
            outside = {
                "track_id": "v-out",
                "score": 0.9,
                "bbox": [0.7 * w, 0.7 * h, 0.9 * w, 0.95 * h],
                "entity_type": "vehicle",
            }
            tagged = tag_detections_with_zones([inside, outside], wh, [zone])
            self.assertEqual(tagged[0].get("zones"), ["no_parking"])
            self.assertNotIn("zones", tagged[1])

    def test_anchor_outside_polygon_extent_is_miss(self) -> None:
        # Polígono solo en la esquina (0..50 px en frame 200x200).
        zone = ZoneConfig(
            id="corner",
            polygon_norm=np.array(
                [[0.0, 0.0], [0.25, 0.0], [0.25, 0.25], [0.0, 0.25]], dtype=np.float64
            ),
        )
        frame_wh = (200, 200)
        poly_abs = denormalize_polygon(zone.polygon_norm, frame_wh)
        x_max, y_max = np.max(poly_abs, axis=0)
        # Anchor BOTTOM_CENTER lejos del extent de la máscara (mask ~ x_max+2).
        far = {
            "track_id": "v-far",
            "score": 0.9,
            "bbox": [150.0, 150.0, 180.0, 190.0],
            "entity_type": "vehicle",
        }
        self.assertGreater(far["bbox"][0], float(x_max) + 2)
        self.assertGreater(far["bbox"][1], float(y_max) + 2)
        tagged = tag_detections_with_zones([far], frame_wh, [zone])
        self.assertNotIn("zones", tagged[0])


class EppZonesAdditiveTests(unittest.TestCase):
    def test_schema_stays_1_0_and_zones_flow(self) -> None:
        self.assertEqual(SCHEMA_VERSION, "1.0")
        norm = _normalize_detection(
            {
                "track_id": "v-1",
                "label": "car",
                "score": 0.9,
                "bbox": [1, 2, 3, 4],
                "zones": ["no_parking"],
                "frame_ts": "2026-07-24T12:00:00Z",
            }
        )
        self.assertEqual(norm["zones"], ["no_parking"])
        events = PerceptionEvent.consolidate_and_emit([norm])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].schema_version, "1.0")
        self.assertEqual(events[0].zones, ["no_parking"])

    def test_rule_zone_no_parking(self) -> None:
        event = PerceptionEventIn(
            entity_type="vehicle",
            confidence=0.5,
            zones=["no_parking"],
            candidate_ids=["track:v-1"],
        )
        self.assertEqual(evaluate_rule(event), "zone:no_parking")


if __name__ == "__main__":
    unittest.main()
