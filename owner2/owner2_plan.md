# Owner 2 Plan: Benchmark & Taxonomy

## Goal

Use Owner 1 manifests to run the core benchmark and expand the work from one pneumonia/WBC experiment into a typed clinical reasoning taxonomy.

## Responsibilities

- Reproduce the kernel benchmark on the Owner 1 pneumonia/WBC manifest.
- Report the four required axes: conflict flips, corroboration controls, directionality, and silent flagging.
- Build the taxonomy cells: contradiction, corroboration, complement, and irrelevance.
- Prototype complement cases where image-only and text-only baselines are near chance, but image+text performs better.
- Maintain benchmark result tables that can feed the shared paper draft.

## Inputs From Owner 1

- `kernel_manifest.csv`: full joined pneumonia/WBC dataset.
- `kernel_480.csv`: balanced reproduction set.
- `kernel_test_1200.csv`: balanced held-out test set if enough cases exist.
- `leakage_report.json`: patient-disjoint split validation.
- `cell_counts.json`: A/B/C/D availability.

## Main Metrics To Report

- Image-only baseline accuracy, sensitivity, and specificity.
- Baseline-validity gate pass/fail.
- Negative flip rate on conflict cells A/B.
- Negative flip rate on control cells C/D.
- Conflict-control gap.
- A-vs-C contrast.
- Abstention or conflict-flagging behavior.

## First Milestone

Reproduce the MedGemma pneumonia/WBC kernel table using the Owner 1 manifest before adding new findings, labs, or models.
