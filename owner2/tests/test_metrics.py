from __future__ import annotations

import unittest

import pandas as pd

from owner2_benchmark.metrics import parse_answer, score_predictions


class ParseAnswerTests(unittest.TestCase):
    def test_parse_answer(self) -> None:
        cases = [
            ("PRESENT", "present"),
            ("The answer is ABSENT.", "absent"),
            ("PRESENT then ABSENT", "absent"),
            ("There is a conflict here.", "conflict"),
            ("unclear", "unparsed"),
            (None, "unparsed"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(parse_answer(raw), expected)


class ScorePredictionTests(unittest.TestCase):
    def test_score_predictions(self) -> None:
        manifest = pd.DataFrame(
            [
                {"subject_id": 1, "study_id": 1, "finding_label": "pos", "cell": "A"},
                {"subject_id": 2, "study_id": 2, "finding_label": "neg", "cell": "B"},
                {"subject_id": 3, "study_id": 3, "finding_label": "pos", "cell": "C"},
                {"subject_id": 4, "study_id": 4, "finding_label": "neg", "cell": "D"},
            ]
        )
        predictions = pd.DataFrame(
            [
                {"subject_id": 1, "study_id": 1, "cond0": "PRESENT", "cond1": "ABSENT", "condA": "CONFLICT"},
                {"subject_id": 2, "study_id": 2, "cond0": "ABSENT", "cond1": "PRESENT", "condA": "CONFLICT"},
                {"subject_id": 3, "study_id": 3, "cond0": "PRESENT", "cond1": "PRESENT", "condA": "PRESENT"},
                {"subject_id": 4, "study_id": 4, "cond0": "ABSENT", "cond1": "ABSENT", "condA": "ABSENT"},
            ]
        )

        scored, metrics = score_predictions(manifest, predictions)

        self.assertEqual(len(scored), 4)
        self.assertTrue(metrics["baseline"]["baseline_valid"])
        self.assertEqual(metrics["negative_flip"]["conflict_A_B"]["numerator"], 2)
        self.assertEqual(metrics["negative_flip"]["conflict_A_B"]["denominator"], 2)
        self.assertEqual(metrics["negative_flip"]["control_C_D"]["numerator"], 0)
        self.assertEqual(metrics["negative_flip"]["gap"], 1.0)
        self.assertEqual(metrics["abstention"]["conflict_usage_overall"]["numerator"], 2)


if __name__ == "__main__":
    unittest.main()
