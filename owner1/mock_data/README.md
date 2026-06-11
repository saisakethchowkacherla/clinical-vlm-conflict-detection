# Mock Dataset

This folder contains tiny synthetic CSV files shaped like the required MIMIC inputs.
It is only for local pipeline testing and does not contain real patient data.

The mock data has 12 valid joined studies:

- 3 cell A cases: pneumonia present + WBC normal/low.
- 3 cell B cases: pneumonia absent + WBC high.
- 3 cell C cases: pneumonia present + WBC high.
- 3 cell D cases: pneumonia absent + WBC normal/low.

Two extra rows are intentionally excluded:

- `10000013`: uncertain/disagreeing label.
- `10000014`: CheXpert/NegBio confident disagreement.

Run from `owner1`:

```powershell
$env:PYTHONPATH='.\src'
python -m owner1_pipeline.cli --config config.mock.toml
```
