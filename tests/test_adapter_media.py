"""Tests stdlib para helpers de auto-select por mtime (adapter).

Ejecutar desde la raíz del repo:

    PYTHONPATH=. python3 tests/test_adapter_media.py
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest

from fastapi.testclient import TestClient

import adapter.app as ad


class PickNewestNameTests(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertIsNone(ad._pick_newest_name({}))

    def test_picks_highest_mtime(self) -> None:
        self.assertEqual(
            ad._pick_newest_name({"a.jpg": 1.0, "b.jpg": 3.0, "c.jpg": 2.0}),
            "b.jpg",
        )

    def test_tie_breaks_lexicographically(self) -> None:
        # max((mtime, name)) → mismo mtime elige el nombre mayor
        self.assertEqual(
            ad._pick_newest_name({"a.jpg": 5.0, "z.jpg": 5.0}),
            "z.jpg",
        )


class MediaChangesDetectedTests(unittest.TestCase):
    def test_new_file(self) -> None:
        self.assertTrue(
            ad._media_changes_detected({"a.jpg": 1.0}, {"a.jpg": 1.0, "b.jpg": 2.0})
        )

    def test_mtime_bump(self) -> None:
        self.assertTrue(
            ad._media_changes_detected({"a.jpg": 1.0}, {"a.jpg": 2.0})
        )

    def test_unchanged(self) -> None:
        self.assertFalse(
            ad._media_changes_detected({"a.jpg": 1.0}, {"a.jpg": 1.0})
        )

    def test_deletion_alone_is_not_change(self) -> None:
        # Solo archivos nuevos/actualizados disparan; borrar no re-selecciona.
        self.assertFalse(
            ad._media_changes_detected({"a.jpg": 1.0, "b.jpg": 2.0}, {"a.jpg": 1.0})
        )


class ScanMediaMtimesTests(unittest.TestCase):
    def test_scans_root_jpgs_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            images = os.path.join(tmp, "images")
            os.makedirs(images)
            nested = os.path.join(images, "nested")
            os.makedirs(nested)
            with open(os.path.join(images, "root.jpg"), "wb") as fh:
                fh.write(b"x")
            with open(os.path.join(nested, "deep.jpg"), "wb") as fh:
                fh.write(b"y")
            with open(os.path.join(images, "note.txt"), "w") as fh:
                fh.write("nope")

            prev_dir = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                mtimes = ad._scan_media_mtimes()
            finally:
                ad.MEDIA_DIR = prev_dir

            self.assertIn("root.jpg", mtimes)
            self.assertNotIn("deep.jpg", mtimes)
            self.assertNotIn("note.txt", mtimes)


class SafeUploadBasenameTests(unittest.TestCase):
    def test_rejects_non_image(self) -> None:
        self.assertIsNone(ad._safe_upload_basename("notes.txt"))
        self.assertIsNone(ad._safe_upload_basename("../etc/passwd.jpg/x"))

    def test_accepts_jpg_and_sanitizes(self) -> None:
        name = ad._safe_upload_basename(r"C:\tmp\my car!!!.JPG")
        self.assertIsNotNone(name)
        assert name is not None
        self.assertTrue(name.endswith(".jpg"))
        self.assertNotIn(" ", name)
        self.assertNotIn("!", name)
        self.assertTrue(re.match(r"^\d{8}_\d{6}_", name))


class MediaOriginalEndpointTests(unittest.TestCase):
    """T1.4: GET /media/original — FileResponse + X-Generation + no-store; 404 sin media."""

    def test_404_without_active_media(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "images"))
            prev_dir = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                with TestClient(ad.app) as client:
                    resp = client.get("/media/original")
                    self.assertEqual(resp.status_code, 404)
            finally:
                ad.MEDIA_DIR = prev_dir

    def test_serves_active_media_with_generation_header_no_store(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "sample.jpg")
        with open(fixture, "rb") as fh:
            fixture_bytes = fh.read()

        with tempfile.TemporaryDirectory() as tmp:
            images = os.path.join(tmp, "images")
            os.makedirs(images)
            with open(os.path.join(images, "sample.jpg"), "wb") as fh:
                fh.write(fixture_bytes)

            prev_dir = ad.MEDIA_DIR
            ad.MEDIA_DIR = tmp
            try:
                with TestClient(ad.app) as client:
                    select_resp = client.post(
                        "/media/select", json={"name": "sample.jpg"}
                    )
                    self.assertEqual(select_resp.status_code, 200)

                    current_resp = client.get("/media/current")
                    expected_generation = current_resp.json()["generation"]

                    resp = client.get("/media/original")
                    self.assertEqual(resp.status_code, 200)
                    self.assertEqual(
                        resp.headers.get("x-generation"), str(expected_generation)
                    )
                    self.assertEqual(resp.headers.get("cache-control"), "no-store")
                    self.assertEqual(resp.content, fixture_bytes)
            finally:
                ad.MEDIA_DIR = prev_dir


class EventsEnvelopeGenerationTests(unittest.TestCase):
    """T1.3: GET /events expone generation + last_ingest_generation (envelope)."""

    def test_empty_events_include_generation_fields(self) -> None:
        with TestClient(ad.app) as client:
            resp = client.get("/events")
            self.assertEqual(resp.status_code, 200)
            body = resp.json()
            self.assertEqual(body["events"], [])
            self.assertEqual(body["count"], 0)
            self.assertEqual(body["generation"], 0)
            self.assertIsNone(body["last_ingest_generation"])


class LastIngestGenerationWiringTests(unittest.TestCase):
    """T1.2: AppState.last_ingest_generation cableado desde trace_id en /ingest."""

    def test_ingest_with_numeric_trace_id_sets_last_ingest_generation(self) -> None:
        with TestClient(ad.app) as client:
            resp = client.post(
                "/ingest",
                json={
                    "trace_id": "3",
                    "detections": [
                        {
                            "track_id": "v-1",
                            "label": "car",
                            "score": 0.9,
                            "entity_type": "vehicle",
                        }
                    ],
                },
            )
            self.assertEqual(resp.status_code, 200)

            events_body = client.get("/events").json()
            self.assertEqual(events_body["last_ingest_generation"], 3)

    def test_ingest_with_non_numeric_trace_id_leaves_last_ingest_generation_none(
        self,
    ) -> None:
        with TestClient(ad.app) as client:
            resp = client.post(
                "/ingest", json={"trace_id": "not-an-int", "detections": []}
            )
            self.assertEqual(resp.status_code, 200)

            events_body = client.get("/events").json()
            self.assertIsNone(events_body["last_ingest_generation"])

    def test_flush_does_not_reset_last_ingest_generation(self) -> None:
        with TestClient(ad.app) as client:
            client.post("/ingest", json={"trace_id": "5", "detections": []})
            self.assertEqual(
                client.get("/events").json()["last_ingest_generation"], 5
            )

            clear_resp = client.post("/media/clear")
            self.assertEqual(clear_resp.status_code, 200)

            # `_flush_detection_session` (invocada por /media/clear) no debe
            # tocar last_ingest_generation — queda stale hasta el próximo
            # /ingest que confirme la nueva generación.
            self.assertEqual(
                client.get("/events").json()["last_ingest_generation"], 5
            )


if __name__ == "__main__":
    unittest.main()
