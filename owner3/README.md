# Owner 3 Models & Scaling

Owner 3 runs models on Owner 1 manifests and emits prediction CSVs for Owner 2 scoring.

Because the real dataset and GPU models are not available yet, this folder includes a mock inference CLI that follows the final output contract.

## Mock Run

From this `owner3` folder:

```powershell
$env:PYTHONPATH='.\src'
python -m owner3_models.cli `
  --manifest ..\owner1\outputs_mock\kernel_manifest.csv `
  --model mock-medgemma-27b `
  --output outputs_mock\mock_medgemma_27b_predictions.csv
```

The output CSV has:

- `subject_id`
- `study_id`
- `model`
- `cond0`
- `cond1`
- `cond2`
- `condA`

Owner 2 can score this file directly.

## Real Model Replacement

Later, replace the mock generator with real model loading while preserving the same CSV contract.
