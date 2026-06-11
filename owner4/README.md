# Owner 4 — Fix (Training): Conflict-Aware LoRA

Builds LoRA training data from the 29,407-study training pool and provides
mock trained models for the fix-ladder evaluation.

## Deliverables
- Training data builder: faithful + defer + naive-consistency examples.
- Leakage check (train pool vs held-out test — must be patient-disjoint).
- Mock conflict-aware LoRA (`MockFixedLoRA`) — simulates a correctly trained fix.
- Mock naive-consistency LoRA (`MockNaiveConsistencyLoRA`) — the "Consistent but
  Dangerous" baseline (low flips everywhere, but text-blind on control cells).
- CLIs for data building and mock inference.

## Input
- `--train-manifest`: the 29,407-study training pool manifest CSV.
- `--test-manifest`: the 1,200-study held-out test manifest CSV (for leakage check).

## Output (data builder)
- `training_faithful.csv` — examples teaching image-faithful answers.
- `training_naive_consistency.csv` — naive baseline examples.
- `training_all.csv` — combined.
- `leakage_report.json` — patient-disjoint confirmation.

## Run data builder on mock manifest

```powershell
$env:PYTHONPATH='.\src'
python -m owner4_training.cli `
  --train-manifest ..\owner1\outputs_mock\kernel_manifest.csv `
  --test-manifest ..\owner1\outputs_mock\kernel_manifest.csv `
  --output-dir outputs_mock
```

## Run mock trained model inference

```powershell
$env:PYTHONPATH='.\src'
python -m owner4_training.cli infer `
  --manifest ..\owner1\outputs_mock\kernel_manifest.csv `
  --output-dir outputs_mock `
  --model both
```

## Model comparison (mock targets)

| Model                  | Cell A NF | NF_conflict | NF_control | CONFLICT usage |
|------------------------|-----------|-------------|------------|----------------|
| Baseline MedGemma-27B  | 57.4%     | 21.8%       | 0.6%       | 0/480          |
| MockFixedLoRA          | ~8%       | ~5%         | ~0.6%      | Uses CONFLICT  |
| MockNaiveLoRA          | ~1%       | ~1%         | ~12%       | Never          |

The naive LoRA demonstrates the "Consistent but Dangerous" trap: near-zero flips
but it ignores the lab on control cells too — text-blind, not truly fixed.
