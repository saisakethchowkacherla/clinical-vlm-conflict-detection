from __future__ import annotations

import unittest

import pandas as pd

from owner5_inference.gates import apply_defer_gate


class GateTests(unittest.TestCase):
    def test_defer_gate_flags_disagreement(self) -> None:
        predictions = pd.DataFrame(
            [
                {"subject_id": 1, "study_id": 1, "model": "m", "cond0": "PRESENT", "cond1": "ABSENT"},
                {"subject_id": 2, "study_id": 2, "model": "m", "cond0": "ABSENT", "cond1": "ABSENT"},
            ]
        )
        gated = apply_defer_gate(predictions)
        self.assertEqual(gated.loc[0, "cond1"], "CONFLICT")
        self.assertEqual(gated.loc[0, "condA"], "CONFLICT")
        self.assertEqual(gated.loc[1, "cond1"], "ABSENT")


if __name__ == "__main__":
    unittest.main()
