# clinical-vlm-conflict-detection
Research on conflict-aware reasoning in medical vision-language models using EHR and chest X-ray data.

## Owner Structure

- `owner1/`: data pipeline and mock MIMIC-shaped dataset.
- `owner2/`: benchmark scorer and taxonomy metrics.
- `owner3/`: model inference contract and mock model generator.
- `owner4/`: training-fix contract and mock LoRA-style fix.
- `owner5/`: inference-time defer gate.
- `owner6/`: rigor, aggregation, and writing summaries.

## Run the Full Mock Pipeline

The real PhysioNet/MIMIC dataset is not required for this demo.

```powershell
.\scripts\run_mock_pipeline.ps1
```

This runs:

1. Owner 1 mock manifest generation.
2. Owner 3 mock model inference.
3. Owner 2 raw benchmark scoring.
4. Owner 5 defer-gate mitigation.
5. Owner 4 mock training mitigation.
6. Owner 6 metrics aggregation.

Generated outputs are ignored by git under each owner's `outputs_mock/` folder.

## Real Data Replacement

When PhysioNet access is approved, replace Owner 1 `config.mock.toml` with a private `config.toml` pointing to local MIMIC files. The downstream owners should keep using the same manifest and prediction CSV contracts.
