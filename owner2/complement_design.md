# Complement Design

## Why This Matters

For the pneumonia/WBC kernel, ignoring the lab is often the safest strategy because the answer is image-grounded. Complement cases are needed to show that both image and text can be genuinely load-bearing.

## Target Property

A valid complement set should satisfy:

- Image-only performance is near chance.
- Text-only performance is near chance.
- Image+text performance is meaningfully better.
- Ground truth comes from an objective future event or external adjudication, not the same report labeler.

## Prototype Ideas

- Ambiguous CXR finding plus future infection oracle:
  - select uncertain or labeler-disagreement pneumonia/consolidation studies;
  - use later positive respiratory culture and antibiotics within 48 hours as an objective signal;
  - include WBC, CRP, or procalcitonin as structured context.
- Etiology disambiguation:
  - use visible opacity, edema, or effusion on CXR;
  - use BNP, albumin, or related labs to help distinguish cause;
  - validate against discharge diagnosis or clinician review.

## Validation Requirement

Before using complement cases as a headline result, compare the automated oracle against a clinician-reviewed sample and report agreement.
