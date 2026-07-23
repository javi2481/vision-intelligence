"""Tests for GET/PUT /capabilities and empty-ingest completeness.

Run from repo root:

    PYTHONPATH=. pytest tests/test_capabilities.py -q
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import adapter.app as ad


SPA_KEYS = {
    "vehicle",
    "object",
    "face",
    "scene",
    "pose",
    "text",
    "face_id",
    "sign",
    "scene_cls",
    "instance",
    "small_object",
    "anomaly",
    "open_vocab",
}


class CapabilitiesGetTests(unittest.TestCase):
    def test_get_spa_keys_booleans_no_side_modules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "images"))
            prev = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                with mock.patch.dict(
                    os.environ,
                    {
                        "ENABLE_FACE_DETECTION": "false",
                        "ENABLE_PEDESTRIAN_ATTRS": "true",
                        "ENABLE_PLATE_OCR": "true",
                    },
                    clear=False,
                ):
                    with TestClient(ad.app) as client:
                        resp = client.get("/capabilities")
                        self.assertEqual(resp.status_code, 200)
                        body = resp.json()
                        self.assertIn("generation", body)
                        caps = body["capabilities"]
                        self.assertEqual(set(caps.keys()), SPA_KEYS)
                        self.assertNotIn("pedestrians", caps)
                        self.assertNotIn("plates", caps)
                        for key, entry in caps.items():
                            self.assertIsInstance(entry["available"], bool)
                            self.assertIsInstance(entry["active"], bool)
                            self.assertIn("name", entry)
                        self.assertTrue(caps["vehicle"]["available"])
                        self.assertTrue(caps["object"]["available"])
                        self.assertTrue(caps["vehicle"]["active"])
                        self.assertTrue(caps["object"]["active"])
                        self.assertFalse(caps["face"]["available"])
                        self.assertFalse(caps["face"]["active"])
                        self.assertTrue(caps["vehicle"].get("critical"))
            finally:
                ad.MEDIA_DIR = prev

    def test_faces_available_when_enable_on(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "images"))
            prev = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                with mock.patch.dict(
                    os.environ, {"ENABLE_FACE_DETECTION": "true"}, clear=False
                ):
                    with TestClient(ad.app) as client:
                        caps = client.get("/capabilities").json()["capabilities"]
                        self.assertTrue(caps["face"]["available"])
                        self.assertTrue(caps["face"]["active"])
            finally:
                ad.MEDIA_DIR = prev


class CapabilitiesPutTests(unittest.TestCase):
    def test_put_merge_vehicle_off_unknown_clamp_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "images"))
            prev = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                with mock.patch.dict(
                    os.environ,
                    {
                        "ENABLE_FACE_DETECTION": "true",
                        "ENABLE_POSE": "false",
                    },
                    clear=False,
                ):
                    with TestClient(ad.app) as client:
                        before = client.get("/capabilities").json()
                        gen0 = before["generation"]
                        self.assertTrue(before["capabilities"]["face"]["active"])

                        # Deactivate faces
                        r1 = client.put(
                            "/capabilities",
                            json={"active": {"face": False}},
                        )
                        self.assertEqual(r1.status_code, 200)
                        body1 = r1.json()
                        self.assertEqual(body1["generation"], gen0 + 1)
                        self.assertFalse(body1["capabilities"]["face"]["active"])
                        self.assertTrue(body1["capabilities"]["vehicle"]["active"])

                        # vehicle.active=false → 400
                        r_v = client.put(
                            "/capabilities",
                            json={"active": {"vehicle": False}},
                        )
                        self.assertEqual(r_v.status_code, 400)

                        # unknown key → 400
                        r_u = client.put(
                            "/capabilities",
                            json={"active": {"nope": True}},
                        )
                        self.assertEqual(r_u.status_code, 400)

                        # activate unavailable (pose) → 400
                        r_p = client.put(
                            "/capabilities",
                            json={"active": {"pose": True}},
                        )
                        self.assertEqual(r_p.status_code, 400)

                        # Re-activate faces; active survives second PUT of other key
                        r2 = client.put(
                            "/capabilities",
                            json={"active": {"face": True}},
                        )
                        self.assertEqual(r2.status_code, 200)
                        self.assertTrue(r2.json()["capabilities"]["face"]["active"])
                        gen_face = r2.json()["generation"]

                        r3 = client.put(
                            "/capabilities",
                            json={"active": {"object": True}},
                        )
                        self.assertEqual(r3.status_code, 200)
                        self.assertEqual(r3.json()["generation"], gen_face + 1)
                        self.assertTrue(r3.json()["capabilities"]["face"]["active"])
                        self.assertTrue(r3.json()["capabilities"]["vehicle"]["active"])
            finally:
                ad.MEDIA_DIR = prev


class EmptyIngestTests(unittest.TestCase):
    def test_empty_ingest_advances_last_ingest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "images"))
            prev = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                with TestClient(ad.app) as client:
                    # Bump generation via PUT (no media required)
                    put = client.put(
                        "/capabilities",
                        json={"active": {"object": True}},
                    )
                    self.assertEqual(put.status_code, 200)
                    gen = put.json()["generation"]
                    self.assertGreater(gen, 0)

                    events_before = client.get("/events").json()
                    self.assertNotEqual(
                        events_before["generation"],
                        events_before["last_ingest_generation"],
                    )

                    ing = client.post(
                        "/ingest",
                        json={"detections": [], "trace_id": gen},
                    )
                    self.assertEqual(ing.status_code, 200)
                    self.assertEqual(ing.json()["accepted"], 0)

                    events = client.get("/events").json()
                    self.assertEqual(events["generation"], gen)
                    self.assertEqual(events["last_ingest_generation"], gen)
            finally:
                ad.MEDIA_DIR = prev


if __name__ == "__main__":
    unittest.main()
