# clinical-vlm-conflict-detection
Research on conflict-aware reasoning in medical vision-language models using chest X-ray and lab data.

## Owner Structure

- `owner1/`: data pipeline — adapts real or synthetic datasets into the kernel manifest.
- `owner2/`: benchmark scorer and taxonomy metrics.
- `owner3/`: model inference contract and mock model generator.
- `owner4/`: training-fix contract and mock LoRA-style fix.
- `owner5/`: inference-time defer gate.
- `owner6/`: rigor, aggregation, and writing summaries.

## Prerequisites

- Python 3.11+
- Install each owner's package (run from the repo root):

```powershell
pip install -e owner1
pip install -e owner2
pip install -e owner3
pip install -e owner4
pip install -e owner5
pip install -e owner6
```

## Option A — COVID-19 Radiography Pipeline (Recommended)

Uses the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
(Kaggle, ~1.1 GB). Download and unzip it so the dataset sits at:

```
data/covid19-radiography-database/COVID-19_Radiography_Dataset/
  COVID/images/
  Normal/images/
  Viral Pneumonia/images/
  Lung_Opacity/images/
```

Then run:

```powershell
.\scripts\run_covid_pipeline.ps1
```

The script automatically converts the dataset into the five pipeline-input CSVs on first run
(`data/covid19/`), then executes all 8 pipeline steps:

| Step | Owner | Action | Output |
|------|-------|--------|--------|
| 0 | — | Prepare COVID CSVs (skipped if already done) | `data/covid19/*.csv` |
| 1 | Owner 1 | Build kernel manifest (480 studies, 120 per cell) | `owner1/outputs_covid/kernel_480.csv` |
| 2 | Owner 3 | Mock MedGemma-27B predictions | `owner3/outputs_covid/` |
| 3 | Owner 2 | Score raw predictions | `owner2/outputs_covid/raw_metrics.json` |
| 4 | Owner 5 | Apply defer-on-disagree gate | `owner5/outputs_covid/` |
| 5 | Owner 2 | Score gated predictions | `owner2/outputs_covid/gate_metrics.json` |
| 6 | Owner 4 | Apply conflict-aware LoRA mock fix | `owner4/outputs_covid/` |
| 7 | Owner 2 | Score LoRA-fixed predictions | `owner2/outputs_covid/lora_metrics.json` |
| 8 | Owner 6 | Aggregate summary | `owner6/outputs_covid/summary.md` |

**Cell structure** (how COVID data maps to the 2×2 conflict kernel):

| Cell | Finding | WBC | Interpretation |
|------|---------|-----|----------------|
| A | COVID (pos) | Normal | Conflict — imaging positive, lab normal |
| B | Normal (neg) | High | Conflict — imaging normal, lab elevated |
| C | COVID (pos) | High | Consistent — both suggest infection |
| D | Normal (neg) | Normal | Consistent — both normal |

## Option B — Synthetic Pipeline (no download needed)

Generates a full-scale synthetic dataset (2,000 studies) and runs the same pipeline steps.

```powershell
python data\synthetic\generate_synthetic_data.py
.\scripts\run_synthetic_pipeline.ps1
```

Outputs go to each owner's `outputs_synthetic/` folder.

## Option C — Small Mock Pipeline

Tiny pre-baked manifest, useful for a quick smoke test.

```powershell
.\scripts\run_mock_pipeline.ps1
```

Outputs go to each owner's `outputs_mock/` folder.

## Demo API

A FastAPI endpoint lets the web demo call the pipeline on a single image:

```powershell
uvicorn api.main:app --port 8000 --reload
```

- `GET  /health` — returns `{"status": "ok"}`
- `POST /predict` — accepts a chest X-ray image + lab value + reference range; returns conflict detection result

## Running Tests

```powershell
python -m pytest owner1/tests owner2/tests owner3/tests owner4/tests owner5/tests owner6/tests
```
