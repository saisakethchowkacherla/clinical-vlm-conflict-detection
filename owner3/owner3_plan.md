# Owner 3 - Models & Scaling

## Goal

Run the benchmark across multiple vision-language models and model sizes, then report which models pass the baseline-validity gate and which show conflict-driven negative flips.

## Core Responsibility

Owner 3 does not build the dataset or scorer. Owner 3 consumes:

- Owner 1 manifests.
- Owner 2 prompt/scoring contract.

Owner 3 produces:

- Raw model outputs for each condition.
- A model gate table.
- Cross-model and cross-scale comparison notes.

## First Models

- `google/medgemma-4b-it`
- `google/medgemma-27b-it`
- `Qwen/Qwen2.5-VL-7B-Instruct`

## First Milestone

Run the image-only baseline condition on the kernel manifest and confirm which models pass the baseline-validity gate before running full conflict experiments.

## Handoff

Raw model outputs should be saved as prediction CSVs that Owner 2 can score directly.
