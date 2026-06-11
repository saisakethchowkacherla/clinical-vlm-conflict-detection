from __future__ import annotations

import unittest

import pandas as pd

from owner4_training.data_builder import build_training_data, leakage_check
from owner4_training.mock_trained_model import make_fixed_lora, make_naive_lora


def _train_manifest() -> pd.DataFrame:
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


def _test_manifest() -> pd.DataFrame:
    return pd.DataFrame([
        {"subject_id": 10, "study_id": 10, "finding_label": "pos", "cell": "A"},
        {"subject_id": 11, "study_id": 11, "finding_label": "neg", "cell": "D"},
    ])


class DataBuilderTests(unittest.TestCase):
    def test_faithful_labels_match_finding(self) -> None:
        df = build_training_data(_train_manifest(), include_naive=False)
        faithful = df[df["example_type"] == "faithful"]
        self.assertEqual(len(faithful), 4)
        for _, row in faithful.iterrows():
            expected = "PRESENT" if row["finding_label"] == "pos" else "ABSENT"
            self.assertEqual(row["label"], expected)

    def test_naive_consistency_labels_match_finding(self) -> None:
        df = build_training_data(_train_manifest(), include_naive=True)
        naive = df[df["example_type"] == "naive_consistency"]
        self.assertEqual(len(naive), 4)
        for _, row in naive.iterrows():
            expected = "PRESENT" if row["finding_label"] == "pos" else "ABSENT"
            self.assertEqual(row["label"], expected)

    def test_defer_examples_label_conflict(self) -> None:
        defer = _train_manifest().copy()
        defer["finding_label"] = "uncertain"
        df = build_training_data(_train_manifest(), defer_manifest=defer, include_naive=False)
        defers = df[df["example_type"] == "defer"]
        self.assertTrue((defers["label"] == "CONFLICT").all())

    def test_leakage_check_passes(self) -> None:
        df = build_training_data(_train_manifest())
        result = leakage_check(df, _test_manifest())
        self.assertTrue(result["leakage_free"])
        self.assertEqual(result["overlap_count"], 0)

    def test_leakage_check_detects_overlap(self) -> None:
        df = build_training_data(_train_manifest())
        # Test manifest shares subject_id=1 with train.
        test = pd.DataFrame([{"subject_id": 1, "study_id": 99, "finding_label": "pos", "cell": "A"}])
        result = leakage_check(df, test)
        self.assertFalse(result["leakage_free"])
        self.assertGreater(result["overlap_count"], 0)


class MockTrainedModelTests(unittest.TestCase):
    def test_fixed_lora_predictions_shape(self) -> None:
        model = make_fixed_lora()
        preds = model.run_manifest_with_labels(_train_manifest())
        self.assertEqual(len(preds), 4)
        for col in ["subject_id", "study_id", "cond0", "cond1", "condA"]:
            self.assertIn(col, preds.columns)

    def test_fixed_lora_conflict_usage_on_flips(self) -> None:
        # Fixed LoRA should output CONFLICT on condA when it flips a conflict cell.
        model = make_fixed_lora(seed=42)
        # Run many cell-A rows to observe CONFLICT usage.
        manifest = pd.DataFrame([
            {"subject_id": i, "study_id": i, "finding_label": "pos", "cell": "A",
             "lab_value": 6.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "normal"}
            for i in range(1, 101)
        ])
        preds = model.run_manifest_with_labels(manifest)
        # Any flip on cell A should produce CONFLICT in condA.
        flips = preds[preds["cond0"] != preds["cond1"]]
        if not flips.empty:
            self.assertTrue((flips["condA"] == "CONFLICT").all())

    def test_naive_lora_never_uses_conflict(self) -> None:
        model = make_naive_lora(seed=42)
        manifest = pd.DataFrame([
            {"subject_id": i, "study_id": i, "finding_label": "pos", "cell": "A",
             "lab_value": 6.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "normal"}
            for i in range(1, 51)
        ])
        preds = model.run_manifest_with_labels(manifest)
        self.assertFalse((preds["condA"] == "CONFLICT").any())

    def test_fixed_lora_lower_flip_rate_than_baseline(self) -> None:
        # Fixed LoRA cell-A flip rate (8%) should be well below baseline (57%).
        model = make_fixed_lora(seed=0)
        manifest = pd.DataFrame([
            {"subject_id": i, "study_id": i, "finding_label": "pos", "cell": "A",
             "lab_value": 6.0, "lab_ref_lower": 4.0, "lab_ref_upper": 11.0, "lab_class": "normal"}
            for i in range(1, 201)
        ])
        preds = model.run_manifest_with_labels(manifest)
        flip_rate = (preds["cond0"] != preds["cond1"]).mean()
        self.assertLess(flip_rate, 0.20)


if __name__ == "__main__":
    unittest.main()
