# Owner 1 Plan: Data & Platform

## Goal

Build the shared data pipeline that converts credentialed PhysioNet/MIMIC files into reproducible manifests for the pneumonia/WBC conflict kernel.

## Datasets

- MIMIC-CXR-JPG: CXR labels, image metadata, split file, and JPG paths.
- MIMIC-IV: `hosp/labevents.csv.gz` for WBC lab values.

## Responsibilities

- Normalize CheXpert and NegBio labels.
- Keep studies where both labelers agree confidently on pneumonia.
- Select one image per study, preferring frontal `PA > AP > other`.
- Join each study to the nearest WBC lab within +/- 24 hours.
- Classify WBC as `high`, `normal`, or `low` using the row reference range.
- Assign A/B/C/D cells for conflict and corroboration.
- Produce patient-disjoint balanced manifests and leakage reports.

## Current Implementation

The pipeline lives in this folder as a standalone Python project:

- `src/owner1_pipeline/pipeline.py`: core data logic.
- `src/owner1_pipeline/cli.py`: command-line entry point.
- `config.example.toml`: local file path template.
- `tests/test_pipeline.py`: unit tests plus synthetic integration test.

## Outputs

- `outputs/kernel_manifest.csv`
- `outputs/kernel_480.csv`
- `outputs/kernel_test_1200.csv`
- `outputs/leakage_report.json`
- `outputs/cell_counts.json`

## Next Steps

1. Get PhysioNet/MIMIC access approved.
2. Copy `config.example.toml` to `config.toml`.
3. Fill in local paths to MIMIC-CXR-JPG and MIMIC-IV files.
4. Run the CLI and verify A/B/C/D counts.
5. Hand the manifest to Owner 2 for benchmark runs.
