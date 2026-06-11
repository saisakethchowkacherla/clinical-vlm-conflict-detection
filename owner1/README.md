# Owner 1 Dataset Pipeline

This repo builds the pneumonia/WBC data manifest for the clinical VLM conflict kernel.
It does not download MIMIC or PhysioNet data. You must provide local credentialed files.

## Inputs

Required local files:

- MIMIC-CXR-JPG CheXpert labels: `mimic-cxr-2.0.0-chexpert.csv.gz`
- MIMIC-CXR-JPG NegBio labels: `mimic-cxr-2.0.0-negbio.csv.gz`
- MIMIC-CXR-JPG metadata: `mimic-cxr-2.0.0-metadata.csv.gz`
- MIMIC-CXR-JPG split file: `mimic-cxr-2.0.0-split.csv.gz`
- MIMIC-IV labs: `hosp/labevents.csv.gz`

## Setup

```powershell
pip install -e .[dev]
```

If `python` is not on PATH, use your available Python executable and run:

```powershell
<python.exe> -m pip install -e .[dev]
```

## Configure

Copy `config.example.toml` to `config.toml` and set absolute paths to your local files.

```powershell
owner1-build --config config.toml
```

Equivalent module form:

```powershell
python -m owner1_pipeline.cli --config config.toml
```

## Outputs

The pipeline writes:

- `outputs/kernel_manifest.csv`: all joined pneumonia/WBC cases.
- `outputs/kernel_480.csv`: balanced 120-per-cell reproduction manifest, if enough cases exist.
- `outputs/kernel_test_1200.csv`: balanced 300-per-cell test manifest, if enough cases exist.
- `outputs/leakage_report.json`: subject overlap checks.
- `outputs/cell_counts.json`: counts before and after sampling.

## What It Does

1. Normalizes CheXpert and NegBio pneumonia labels.
2. Keeps studies where both labelers agree confidently (`pos` or `neg`).
3. Loads WBC labs using item IDs `51301` and `51300`.
4. Joins each study to the nearest WBC within +/- 24 hours.
5. Classifies WBC using the row's own reference range.
6. Assigns cells A/B/C/D.
7. Chooses one image per study, preferring `PA > AP > other`.
8. Creates patient-disjoint balanced manifests.

## Test Without MIMIC

The tests use tiny synthetic CSVs, so you can verify the code without MIMIC access:

```powershell
pytest
```

## Run With Mock Data

This folder includes a tiny synthetic dataset in `mock_data/`.
It is not real patient data; it only exists to test the pipeline shape.

```powershell
python -m owner1_pipeline.cli --config config.mock.toml
```

If the package is not installed yet, run from this `owner1` folder with:

```powershell
$env:PYTHONPATH='.\src'
python -m owner1_pipeline.cli --config config.mock.toml
```
