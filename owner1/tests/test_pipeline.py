from __future__ import annotations

import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from owner1_pipeline.pipeline import (
    PipelineConfig,
    assign_cell,
    classify_lab_value,
    join_nearest_lab,
    normalize_label,
    run_pipeline,
)


def write_csv_gz(path: Path, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)


class PipelineUnitTests(unittest.TestCase):
    def test_normalize_label(self) -> None:
        cases = [
            (1, "pos"),
            ("1.0", "pos"),
            (0, "neg"),
            ("0.0", "neg"),
            (-1, "uncertain"),
            ("-1.0", "uncertain"),
            ("", "na"),
            (None, "na"),
            (float("nan"), "na"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_label(raw), expected)

    def test_classify_lab_value(self) -> None:
        self.assertEqual(classify_lab_value(12.0, 4.0, 11.0), "high")
        self.assertEqual(classify_lab_value(3.9, 4.0, 11.0), "low")
        self.assertEqual(classify_lab_value(7.0, 4.0, 11.0), "normal")
        self.assertEqual(classify_lab_value(4.0, 4.0, 11.0), "normal")
        self.assertEqual(classify_lab_value(11.0, 4.0, 11.0), "normal")

    def test_assign_cell(self) -> None:
        cases = [
            ("pos", "normal", "A"),
            ("pos", "low", "A"),
            ("neg", "high", "B"),
            ("pos", "high", "C"),
            ("neg", "normal", "D"),
            ("neg", "low", "D"),
        ]
        for finding_label, lab_class, cell in cases:
            with self.subTest(finding_label=finding_label, lab_class=lab_class):
                self.assertEqual(assign_cell(finding_label, lab_class), cell)

    def test_join_nearest_lab_inside_window(self) -> None:
        studies = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "study_id": 10,
                    "study_datetime": pd.Timestamp("2200-01-01 12:00:00"),
                }
            ]
        )
        labs = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "itemid": 51301,
                    "charttime": pd.Timestamp("2200-01-01 08:00:00"),
                    "valuenum": 20.0,
                    "ref_range_lower": 4.0,
                    "ref_range_upper": 11.0,
                    "lab_class": "high",
                },
                {
                    "subject_id": 1,
                    "itemid": 51301,
                    "charttime": pd.Timestamp("2200-01-01 13:00:00"),
                    "valuenum": 7.0,
                    "ref_range_lower": 4.0,
                    "ref_range_upper": 11.0,
                    "lab_class": "normal",
                },
            ]
        )

        joined = join_nearest_lab(studies, labs, window_hours=24)

        self.assertEqual(len(joined), 1)
        self.assertEqual(joined.loc[0, "lab_value"], 7.0)
        self.assertEqual(joined.loc[0, "lab_delta_hours"], 1.0)

    def test_join_nearest_lab_outside_window_drops_study(self) -> None:
        studies = pd.DataFrame(
            [{"subject_id": 1, "study_id": 10, "study_datetime": pd.Timestamp("2200-01-01")}]
        )
        labs = pd.DataFrame(
            [
                {
                    "subject_id": 1,
                    "itemid": 51301,
                    "charttime": pd.Timestamp("2200-01-03"),
                    "valuenum": 7.0,
                    "ref_range_lower": 4.0,
                    "ref_range_upper": 11.0,
                    "lab_class": "normal",
                }
            ]
        )

        self.assertTrue(join_nearest_lab(studies, labs, window_hours=24).empty)


class PipelineIntegrationTests(unittest.TestCase):
    def test_run_pipeline_with_synthetic_files(self) -> None:
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            chexpert = pd.DataFrame(
                [
                    {"subject_id": 10000001, "study_id": 5001, "Pneumonia": 1},
                    {"subject_id": 10000002, "study_id": 5002, "Pneumonia": 0},
                    {"subject_id": 10000003, "study_id": 5003, "Pneumonia": 1},
                    {"subject_id": 10000004, "study_id": 5004, "Pneumonia": 0},
                    {"subject_id": 10000005, "study_id": 5005, "Pneumonia": -1},
                ]
            )
            negbio = pd.DataFrame(
                [
                    {"subject_id": 10000001, "study_id": 5001, "Pneumonia": 1},
                    {"subject_id": 10000002, "study_id": 5002, "Pneumonia": 0},
                    {"subject_id": 10000003, "study_id": 5003, "Pneumonia": 1},
                    {"subject_id": 10000004, "study_id": 5004, "Pneumonia": 0},
                    {"subject_id": 10000005, "study_id": 5005, "Pneumonia": 1},
                ]
            )
            metadata = pd.DataFrame(
                [
                    {
                        "dicom_id": "d1",
                        "subject_id": 10000001,
                        "study_id": 5001,
                        "ViewPosition": "AP",
                        "StudyDate": 22000101,
                        "StudyTime": 120000,
                    },
                    {
                        "dicom_id": "d1pa",
                        "subject_id": 10000001,
                        "study_id": 5001,
                        "ViewPosition": "PA",
                        "StudyDate": 22000101,
                        "StudyTime": 120000,
                    },
                    {
                        "dicom_id": "d2",
                        "subject_id": 10000002,
                        "study_id": 5002,
                        "ViewPosition": "AP",
                        "StudyDate": 22000101,
                        "StudyTime": 120000,
                    },
                    {
                        "dicom_id": "d3",
                        "subject_id": 10000003,
                        "study_id": 5003,
                        "ViewPosition": "AP",
                        "StudyDate": 22000101,
                        "StudyTime": 120000,
                    },
                    {
                        "dicom_id": "d4",
                        "subject_id": 10000004,
                        "study_id": 5004,
                        "ViewPosition": "AP",
                        "StudyDate": 22000101,
                        "StudyTime": 120000,
                    },
                ]
            )
            split = metadata[["dicom_id", "subject_id", "study_id"]].copy()
            split["split"] = "test"
            labevents = pd.DataFrame(
                [
                    {
                        "subject_id": 10000001,
                        "itemid": 51301,
                        "charttime": "2200-01-01 11:00:00",
                        "valuenum": 7.0,
                        "ref_range_lower": 4.0,
                        "ref_range_upper": 11.0,
                    },
                    {
                        "subject_id": 10000002,
                        "itemid": 51301,
                        "charttime": "2200-01-01 13:00:00",
                        "valuenum": 12.0,
                        "ref_range_lower": 4.0,
                        "ref_range_upper": 11.0,
                    },
                    {
                        "subject_id": 10000003,
                        "itemid": 51300,
                        "charttime": "2200-01-01 12:30:00",
                        "valuenum": 14.0,
                        "ref_range_lower": 4.0,
                        "ref_range_upper": 11.0,
                    },
                    {
                        "subject_id": 10000004,
                        "itemid": 51301,
                        "charttime": "2200-01-01 12:45:00",
                        "valuenum": 8.0,
                        "ref_range_lower": 4.0,
                        "ref_range_upper": 11.0,
                    },
                ]
            )

            paths = {}
            for name, frame in {
                "chexpert": chexpert,
                "negbio": negbio,
                "metadata": metadata,
                "split": split,
                "labevents": labevents,
            }.items():
                paths[name] = tmp_path / f"{name}.csv.gz"
                write_csv_gz(paths[name], frame)

            config = PipelineConfig(
                chexpert_labels=paths["chexpert"],
                negbio_labels=paths["negbio"],
                metadata=paths["metadata"],
                split=paths["split"],
                labevents=paths["labevents"],
                output_dir=tmp_path / "outputs",
                kernel_per_cell=1,
                test_per_cell=1,
            )

            summary = run_pipeline(config)
            manifest = pd.read_csv(tmp_path / "outputs" / "kernel_manifest.csv")

            self.assertEqual(summary["rows"], 4)
            self.assertEqual(set(manifest["cell"]), {"A", "B", "C", "D"})
            selected_subject1 = manifest.loc[manifest["subject_id"] == 10000001].iloc[0]
            self.assertEqual(selected_subject1["dicom_id"], "d1pa")
            self.assertTrue(Path(tmp_path / "outputs" / "kernel_480.csv").exists())
            self.assertTrue(Path(tmp_path / "outputs" / "cell_counts.json").exists())
            self.assertTrue(Path(tmp_path / "outputs" / "leakage_report.json").exists())


if __name__ == "__main__":
    unittest.main()
