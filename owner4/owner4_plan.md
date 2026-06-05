# Owner 4 - Fix (Training)

## Goal

Develop training-based fixes that reduce conflict-driven negative flips without making the model ignore useful structured text.

## Core Responsibility

Owner 4 consumes:

- Owner 1 train pool and held-out manifests.
- Owner 2 scoring contract.
- Owner 3 model outputs or model-loading setup.

Owner 4 produces:

- Training datasets for conflict-aware fine-tuning.
- LoRA or DPO/RLVR experiments.
- A comparison against naive consistency training.

## First Milestone

Create a small LoRA smoke-test dataset from the train pool and verify that training/evaluation can run end-to-end on a tiny sample.

## Success Criteria

A training fix is only useful if it:

- reduces conflict negative flips by at least 50%;
- does not reduce image-only accuracy;
- does not become text-blind;
- still uses text on complement/text-relevant cases;
- beats or meaningfully complements the simple inference gate from Owner 5.
