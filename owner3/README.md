# Owner 3 — Models & Scaling

Runs the inference harness across multiple VLMs, applies the baseline-validity gate,
and produces a cross-model/cross-scale conflict matrix.

## Deliverables
- Inference contract (`VLMInferenceModel`) every model adapter must implement.
- Mock model generator reproducing the validated MedGemma-27B flip profile.
- Baseline-validity gate (EXPERIMENT_SPEC §4).
- CLI to run all mock models and emit a gate table + prediction CSVs.

## Input
- Owner 1 manifest CSV (`kernel_480.csv` or `kernel_manifest.csv`).

## Output
- `predictions_<model>.csv` — one per model, columns: `subject_id, study_id, cond0, cond1, condA`.
- `gate_table.csv` — baseline gate result per model.
- `gate_summary.json` — same data as JSON.

## Run on mock manifest

```powershell
$env:PYTHONPATH='.\src'
python -m owner3_models.cli `
  --manifest ..\owner1\outputs_mock\kernel_manifest.csv `
  --output-dir outputs_mock
```

## Models available (mock)

| Model name            | NF_conflict | NF_control | Notes                        |
|-----------------------|-------------|------------|------------------------------|
| mock-medgemma-4b      | ~13.7%      | ~0.6%      | Matches validated 4B result  |
| mock-medgemma-27b     | ~21.8%      | ~0.6%      | Matches validated 27B result |
| mock-fixed-lora       | ~5%         | ~0.6%      | Post-training fix (Owner 4)  |
| mock-naive-lora       | ~1%         | ~12%       | Naive baseline — text-blind  |

## Adding a real model adapter
Subclass `VLMInferenceModel` from `owner3_models.contract` and implement `predict_row`.
The `run_manifest` method is inherited and handles the CSV loop.
