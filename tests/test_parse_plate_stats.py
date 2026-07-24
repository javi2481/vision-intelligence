"""Unit tests for parse_plate counters (PR1 tiling baseline)."""

from __future__ import annotations

import unittest

from detection.plates.client import (
    parse_plate,
    plate_parse_stats,
    reset_plate_parse_stats,
)


class ParsePlateStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_plate_parse_stats()

    def test_counts_rejected_and_accepted(self) -> None:
        best = parse_plate(
            ["AB!", "ABC123", "TOOLONGPLATE99", "XY9876"],
            [0.2, 0.9, 0.5, 0.8],
        )
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best["text"], "ABC123")
        stats = plate_parse_stats()
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["rejected_regex"], 2)  # AB! cleaned may fail length; TOOLONG
        self.assertEqual(stats["accepted"], 2)

    def test_empty_inputs(self) -> None:
        self.assertIsNone(parse_plate([], []))
        self.assertEqual(
            plate_parse_stats(),
            {"total": 0, "rejected_regex": 0, "accepted": 0},
        )


if __name__ == "__main__":
    unittest.main()
