# Training Data Plan

## Inputs

Use Owner 1 training pool once available:

- patient-disjoint from held-out test;
- includes image path, finding label, WBC value, WBC class, and cell.

## Dataset Types

### Faithful Image-Answer Examples

Use A/B/C/D cases where the target answer follows the image label:

- image positive -> `PRESENT`
- image negative -> `ABSENT`

This teaches the model not to abandon clear image evidence because of a conflicting lab.

### Defer Examples

Use uncertain or ambiguous cases where the right behavior should be `CONFLICT` or defer.

Do not use cross-labeler disagreement as image ground truth. Use it only as a signal that the case may need deferral or a future-event oracle.

### Naive Consistency Baseline

Train a baseline that simply enforces agreement between image-only and image+lab answers.

This baseline is important because it may reduce flips by becoming text-blind; the real fix must beat it.

## Splitting Rule

No subject from the training data may appear in the held-out Owner 1 test manifest.
