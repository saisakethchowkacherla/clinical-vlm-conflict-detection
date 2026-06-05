# Owner 6 Rigor, Stats & Writing

Owner 6 gathers scored metrics from Owner 2, checks the headline fields, and creates compact summary tables for writing.

## Aggregate Mock Results

From this `owner6` folder:

```powershell
$env:PYTHONPATH='.\src'
python -m owner6_rigor.cli `
  --metrics ..\owner2\outputs_mock\metrics.json `
  --output outputs_mock\summary.md
```

Pass multiple `--metrics` files to compare raw, gated, and training-fix results.
