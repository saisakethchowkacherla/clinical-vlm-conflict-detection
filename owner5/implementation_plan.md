# Implementation Plan

## Input Needed

- image-only prediction (cond0)
- image + lab prediction (cond1)
- true label

## Conflict Detection

If cond0 != cond1:
- flag as CONFLICT

## Evaluation

- Accuracy
- Flip Rate
- Conflict Rate

## Output

Compare:
- Raw Model
- Image Only
- Defer Gate
