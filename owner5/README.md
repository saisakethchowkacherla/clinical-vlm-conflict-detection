# Owner 5 Inference Fix & Frontier

Owner 5 applies training-free fixes to model prediction CSVs.

## Defer Gate

If `cond0` and `cond1` disagree, replace the lab-conditioned answer with `CONFLICT`.

This preserves the model's image-only answer while flagging cases where the lab-conditioned prompt changed the answer.

## Mock Run

From this `owner5` folder:

```powershell
$env:PYTHONPATH='.\src'
python -m owner5_inference.cli `
  --predictions ..\owner3\outputs_mock\mock_medgemma_27b_predictions.csv `
  --strategy defer-on-disagree `
  --output outputs_mock\defer_gate_predictions.csv
```

Score the output with Owner 2.
