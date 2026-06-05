# LoRA Plan

## Starting Model

Begin with a smaller open model:

- `google/medgemma-4b-it`

Only attempt 27B training after the 4B training loop is proven.

## Training Setup

- Use LoRA adapters.
- Keep base model frozen.
- Start with a tiny smoke run before full training.
- Save all adapter checkpoints outside committed source folders.

## Minimal Smoke Test

1. Build 20-50 synthetic or mock-style examples.
2. Run a few training steps.
3. Confirm loss decreases or training completes.
4. Run Owner 2 scorer on a tiny held-out mock output.

## Full Training

After smoke test:

1. Train on the leakage-safe train pool.
2. Evaluate on `kernel_480.csv`.
3. Evaluate on `kernel_test_1200.csv`.
4. Compare against image-only, no-fix, Owner 5 gate, and naive consistency LoRA.

## Risks

- The LoRA may suppress useful text.
- The LoRA may overfit the pneumonia/WBC pair.
- The LoRA may reduce image-only accuracy.
- The simple inference gate may outperform training.
