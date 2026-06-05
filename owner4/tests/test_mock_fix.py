from __future__ import annotations

import unittest

import pandas as pd

from owner4_training.mock_fix import apply_mock_training_fix


class MockFixTests(unittest.TestCase):
    def test_conflict_aware_fix_restores_cond0_and_flags_conflict(self) -> None:
        predictions = pd.DataFrame(
            [
                {"subject_id": 1, "study_id": 1, "model": "m", "cond0": "PRESENT", "cond1": "ABSENT"},
                {"subject_id": 2, "study_id": 2, "model": "m", "cond0": "ABSENT", "cond1": "ABSENT"},
            ]
        )
        fixed = apply_mock_training_fix(predictions, "conflict-aware-lora-mock")
        self.assertEqual(fixed.loc[0, "cond1"], "PRESENT")
        self.assertEqual(fixed.loc[0, "condA"], "CONFLICT")
        self.assertEqual(fixed.loc[1, "cond1"], "ABSENT")


if __name__ == "__main__":
    unittest.main()
