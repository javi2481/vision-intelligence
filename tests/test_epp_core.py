"""Test standalone para adapter.epp_core (contrato PerceptionEvent).

Ejecutar desde la raíz del repo:

    PYTHONPATH=. python3 tests/test_epp_core.py
"""

import unittest

from adapter.epp_core import PerceptionEvent, _normalize_detection


class NormalizeDetectionEntityTypeTests(unittest.TestCase):
    def test_defaults_to_vehicle_when_missing(self) -> None:
        normalized = _normalize_detection({"track_id": "1", "label": "car", "score": 0.9})
        self.assertEqual(normalized["entity_type"], "vehicle")

    def test_passes_through_object_entity_type(self) -> None:
        normalized = _normalize_detection(
            {"track_id": "o-1", "label": "person", "score": 0.8, "entity_type": "object"}
        )
        self.assertEqual(normalized["entity_type"], "object")


class ConsolidateAndEmitVehicleRegressionTests(unittest.TestCase):
    """Comportamiento de vehicle_type/color/plate_text sin cambios (regresión)."""

    def test_vehicle_track_votes_color_type_plate(self) -> None:
        detections = [
            {
                "track_id": "v-1",
                "label": "sedan",
                "score": 0.9,
                "color": "white",
                "plate": {"text": "ABC123", "score": 0.85},
                "bbox": [0, 0, 10, 10],
                "frame_ts": "2026-07-18T15:00:00Z",
            },
            {
                "track_id": "v-1",
                "label": "sedan",
                "score": 0.85,
                "color": "white",
                "plate": {"text": "ABC123", "score": 0.8},
                "bbox": [0, 0, 10, 10],
                "frame_ts": "2026-07-18T15:00:01Z",
            },
        ]

        events = PerceptionEvent.consolidate_and_emit(detections)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.entity_type, "vehicle")
        self.assertEqual(event.payload.vehicle_type, "sedan")
        self.assertEqual(event.payload.color, "white")
        self.assertEqual(event.payload.plate_text, "ABC123")
        self.assertIsNotNone(event.payload.plate_confidence)
        self.assertIsNone(event.payload.class_name)
        self.assertIn("patente:ABC123", event.candidate_ids)


class ConsolidateAndEmitObjectTrackTests(unittest.TestCase):
    def test_object_track_votes_class_name_no_color_no_plate(self) -> None:
        detections = [
            {
                "track_id": "o-1",
                "label": "person",
                "score": 0.9,
                "bbox": [5, 5, 20, 40],
                "entity_type": "object",
                "frame_ts": "2026-07-18T15:00:00Z",
            },
            {
                "track_id": "o-1",
                "label": "person",
                "score": 0.8,
                "bbox": [5, 5, 20, 40],
                "entity_type": "object",
                "frame_ts": "2026-07-18T15:00:01Z",
            },
        ]

        events = PerceptionEvent.consolidate_and_emit(detections)

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.entity_type, "object")
        self.assertEqual(event.payload.class_name, "person")
        self.assertIsNone(event.payload.color)
        self.assertIsNone(event.payload.plate_text)
        self.assertIsNone(event.payload.plate_confidence)
        self.assertIsNone(event.payload.vehicle_type)
        self.assertNotIn("patente:", "".join(event.candidate_ids))
        self.assertIn("track:o-1", event.candidate_ids)

    def test_object_track_does_not_crash_without_score_or_bbox(self) -> None:
        detections = [
            {
                "track_id": "o-2",
                "label": "dog",
                "score": 0.5,
                "entity_type": "object",
            }
        ]

        events = PerceptionEvent.consolidate_and_emit(detections)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload.class_name, "dog")


class ConsolidateAndEmitFaceAndSceneTests(unittest.TestCase):
    def test_face_track(self) -> None:
        events = PerceptionEvent.consolidate_and_emit(
            [
                {
                    "track_id": "f-1",
                    "label": "face",
                    "score": 0.92,
                    "bbox": [1, 2, 3, 4],
                    "entity_type": "face",
                    "frame_ts": "2026-07-18T15:00:00Z",
                }
            ]
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].entity_type, "face")
        self.assertEqual(events[0].payload.class_name, "face")

    def test_scene_track(self) -> None:
        events = PerceptionEvent.consolidate_and_emit(
            [
                {
                    "track_id": "scene-0",
                    "label": "street",
                    "score": 0.7,
                    "bbox": [0, 0, 100, 80],
                    "entity_type": "scene",
                    "scene": {
                        "type": "street",
                        "ratios": {"road": 0.3},
                        "infra": {"has_road": True},
                        "lanes": None,
                        "crosswalk": None,
                    },
                    "frame_ts": "2026-07-18T15:00:00Z",
                }
            ]
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].entity_type, "scene")
        self.assertEqual(events[0].payload.scene_type, "street")
        self.assertEqual(events[0].payload.class_name, "street")
        self.assertIn("scene:street", events[0].candidate_ids)

    def test_person_with_attrs(self) -> None:
        events = PerceptionEvent.consolidate_and_emit(
            [
                {
                    "track_id": "o-9",
                    "label": "person",
                    "score": 0.88,
                    "bbox": [1, 2, 3, 4],
                    "entity_type": "object",
                    "person": {"gender": "female", "age_group": "adult"},
                    "frame_ts": "2026-07-18T15:00:00Z",
                }
            ]
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].payload.person["gender"], "female")

    def test_pose_and_text_tracks(self) -> None:
        events = PerceptionEvent.consolidate_and_emit(
            [
                {
                    "track_id": "k-1",
                    "label": "person_pose",
                    "score": 0.8,
                    "bbox": [1, 2, 3, 4],
                    "entity_type": "pose",
                    "keypoints": [[1, 2]],
                    "frame_ts": "2026-07-18T15:00:00Z",
                },
                {
                    "track_id": "t-0",
                    "label": "text",
                    "score": 0.9,
                    "bbox": [1, 2, 3, 4],
                    "entity_type": "text",
                    "text": "STOP",
                    "frame_ts": "2026-07-18T15:00:00Z",
                },
            ]
        )
        types = {e.entity_type for e in events}
        self.assertEqual(types, {"pose", "text"})
        text_ev = next(e for e in events if e.entity_type == "text")
        self.assertEqual(text_ev.payload.text, "STOP")


if __name__ == "__main__":
    unittest.main()
