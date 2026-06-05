# Owner 2 Benchmark & Taxonomy

Owner 2 consumes Owner 1 manifests and model output files, then computes the benchmark metrics for the pneumonia/WBC kernel.

## Input Files

### Manifest

The manifest comes from Owner 1 and must include:

- `subject_id`
- `study_id`
- `finding_label` with values `pos` or `neg`
- `cell` with values `A`, `B`, `C`, `D`

### Predictions

Predictions are a CSV with one row per `study_id` and one column per condition:

```csv
subject_id,study_id,cond0,cond1,condA
10000001,5001,PRESENT,ABSENT,CONFLICT
```

Required columns:

- `cond0`: image-only output.
- `cond1`: image + plain lab output.

Optional column:

- `condA`: image + lab + explicit conflict option.

## Run On Mock Predictions

From this `owner2` folder:

```powershell
$env:PYTHONPATH='.\src'
python -m owner2_benchmark.cli `
  --manifest ..\owner1\outputs_mock\kernel_manifest.csv `
  --predictions mock_predictions\predictions.csv `
  --output outputs_mock\metrics.json
```

If `python` is not on PATH, use the bundled Python executable used by Codex.

## Main Metrics

- Image-only accuracy, sensitivity, and specificity.
- Baseline-validity gate.
- Negative flip rate on conflict cells A/B.
- Negative flip rate on control cells C/D.
- Conflict-control gap.
- A-vs-C contrast.
- Optional `CONFLICT` usage for `condA`.
