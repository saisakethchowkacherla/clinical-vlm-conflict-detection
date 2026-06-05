# Evaluation Guardrails

## Must Report

- Image-only accuracy before and after training.
- Negative flip rate on conflict cells.
- Negative flip rate on control cells.
- Conflict-control gap.
- A-vs-C contrast.
- Abstention or conflict usage.
- Text-use preservation on complement cases once available.

## Do Not Claim Success If

- Flip rate drops only because the model always ignores text.
- Image-only accuracy drops.
- The fix only works on the training pair and fails out of distribution.
- The trained model fails the baseline-validity gate.

## Required Baselines

- Original model, no fix.
- Image-only answer.
- Owner 5 defer-on-disagree gate.
- Naive consistency LoRA.
