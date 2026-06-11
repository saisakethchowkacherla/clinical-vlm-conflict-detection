from __future__ import annotations

import unittest

import pandas as pd

from owner3_models.gate import run_baseline_gate
from owner3_models.mock_model import make_mock_model


def _manifest() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": 1, "study_id": 1, "finding_label": "pos", "cell": "A",
         "lab_value": 6.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "normal"},
        {"subject_id": 2, "study_id": 2, "finding_label": "neg", "cell": "B",
         "lab_value": 14.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "high"},
        {"subject_id": 3, "study_id": 3, "finding_label": "pos", "cell": "C",
         "lab_value": 14.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "high"},
        {"subject_id": 4, "study_id": 4, "finding_label": "neg", "cell": "D",
         "lab_value": 6.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "normal"},
    ])


class MockModelTests(unittest.TestCase):
    def test_predictions_shape(self) -> None:
        model = make_mock_model(seed=42)
        manifest = _manifest()
        preds = model.run_manifest_with_labels(manifest)
        self.assertEqual(len(preds), 4)
        for col in ["subject_id", "study_id", "cond0", "cond1", "condA"]:
            self.assertIn(col, preds.columns)

    def test_cond0_matches_finding_label(self) -> None:
        model = make_mock_model(seed=42)
        manifest = _manifest()
        preds = model.run_manifest_with_labels(manifest)
        merged = manifest.merge(preds, on=["subject_id", "study_id"])
        for _, row in merged.iterrows():
            expected = "PRESENT" if row["finding_label"] == "pos" else "ABSENT"
            self.assertEqual(row["cond0"], expected)

    def test_cell_d_never_flips(self) -> None:
        # Cell D flip rate is 0.0 — cond1 must always equal cond0.
        model = make_mock_model(seed=99)
        manifest = pd.DataFrame([
            {"subject_id": i, "study_id": i, "finding_label": "neg", "cell": "D",
             "lab_value": 6.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "normal"}
            for i in range(1, 51)
        ])
        preds = model.run_manifest_with_labels(manifest)
        self.assertTrue((preds["cond0"] == preds["cond1"]).all())

    def test_deterministic_with_same_seed(self) -> None:
        manifest = _manifest()
        preds1 = make_mock_model(seed=7).run_manifest_with_labels(manifest)
        preds2 = make_mock_model(seed=7).run_manifest_with_labels(manifest)
        pd.testing.assert_frame_equal(preds1, preds2)


class BaselineGateTests(unittest.TestCase):
    def test_valid_model_passes(self) -> None:
        manifest = _manifest()
        model = make_mock_model(seed=42)
        preds = model.run_manifest_with_labels(manifest)
        gate = run_baseline_gate(preds, manifest)
        # With cond0 = perfect image answers, gate must pass.
        self.assertTrue(gate["baseline_valid"])

    def test_degenerate_model_fails(self) -> None:
        manifest = _manifest()
        # Model that always says ABSENT regardless.
        preds = manifest[["subject_id", "study_id"]].copy()
        preds["cond0"] = "ABSENT"
        gate = run_baseline_gate(preds, manifest)
        # sensitivity = 0 (never predicts PRESENT for pos) → degenerate.
        self.assertFalse(gate["baseline_valid"])


if __name__ == "__main__":
    unittest.main()
