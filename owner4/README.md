did # Owner 4 Training Fix

Owner 4 develops training-based mitigation methods. Without the real dataset/GPU, this folder implements a mock fix that transforms prediction CSVs while preserving the final evaluation contract.

## Mock Fix Run

From this `owner4` folder:

```powershell
$env:PYTHONPATH='.\src'
python -m owner4_training.cli `
  --predictions ..\owner3\outputs_mock\mock_medgemma_27b_predictions.csv `
  --strategy conflict-aware-lora-mock `
  --output outputs_mock\mock_lora_predictions.csv
```

Then score the output with Owner 2.

## Real Replacement

Later, real LoRA/DPO training should produce prediction CSVs with the same columns as Owner 3:

- `subject_id`
- `study_id`
- `model`
- `cond0`
- `cond1`
- `cond2`
- `condA`
