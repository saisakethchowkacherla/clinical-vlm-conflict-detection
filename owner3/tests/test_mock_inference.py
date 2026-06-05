from __future__ import annotations

import unittest

import pandas as pd

from owner3_models.mock_inference import PROFILES, generate_mock_predictions


class MockInferenceTests(unittest.TestCase):
    def test_mock_medgemma_27b_flips_conflict_cells(self) -> None:
        manifest = pd.DataFrame(
            [
                {"subject_id": 1, "study_id": 1, "finding_label": "pos", "cell": "A"},
                {"subject_id": 2, "study_id": 2, "finding_label": "neg", "cell": "B"},
                {"subject_id": 3, "study_id": 3, "finding_label": "pos", "cell": "C"},
                {"subject_id": 4, "study_id": 4, "finding_label": "neg", "cell": "D"},
            ]
        )
        predictions = generate_mock_predictions(manifest, PROFILES["mock-medgemma-27b"])
        self.assertEqual(list(predictions["cond0"]), ["PRESENT", "ABSENT", "PRESENT", "ABSENT"])
        self.assertEqual(list(predictions["cond1"]), ["ABSENT", "PRESENT", "PRESENT", "ABSENT"])

    def test_qwen_profile_is_degenerate_absent(self) -> None:
        manifest = pd.DataFrame(
            [
                {"subject_id": 1, "study_id": 1, "finding_label": "pos", "cell": "A"},
                {"subject_id": 2, "study_id": 2, "finding_label": "neg", "cell": "D"},
            ]
        )
        predictions = generate_mock_predictions(manifest, PROFILES["mock-qwen-degenerate"])
        self.assertEqual(list(predictions["cond0"]), ["ABSENT", "ABSENT"])


if __name__ == "__main__":
    unittest.main()
