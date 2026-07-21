"""Tests stdlib para helpers de auto-select por mtime (adapter).

Ejecutar: python test_adapter_media.py
"""

from __future__ import annotations

import os
import re
import tempfile
import unittest

import adapter as ad


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


if __name__ == "__main__":
    unittest.main()
