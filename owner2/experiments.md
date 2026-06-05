# Owner 2 Experiments

## Experiment 1: Kernel Reproduction

Use the Owner 1 `kernel_480.csv` manifest and run:

- image-only prompt;
- image plus WBC prompt;
- image plus WBC plus clinical-lean prompt as secondary;
- image plus WBC with explicit `CONFLICT` option.

Primary target: reproduce the MedGemma-27B pneumonia/WBC numbers from the experiment spec.

## Experiment 2: Held-Out Test

Use `kernel_test_1200.csv` if Owner 1 can build enough patient-disjoint cases.

Report the same metrics as the kernel reproduction:

- baseline gate;
- negative flip rate;
- conflict-control gap;
- A-vs-C contrast;
- abstention usage.

## Experiment 3: Taxonomy Expansion

After the kernel reproduces, add candidate finding/lab pairs only after Owner 1 provides pre-counts and clinical sanity checks.

Candidate pairs:

- Consolidation/WBC
- Consolidation/CRP
- Edema/BNP
- Effusion/BNP

## Experiment 4: Complement Prototype

Build one small complement prototype and run unimodal baselines before treating it as a benchmark cell.
