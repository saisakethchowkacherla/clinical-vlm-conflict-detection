# Defer Gate

## Goal

Detect disagreement between image-only and image+lab predictions.

## Rule

If image_only != image_lab

Output = CONFLICT

Else

Output = image_lab

## Benefits

- Identifies uncertain cases
- Reduces incorrect decisions
- Improves reliability

## Experimental Result

- Raw Accuracy: 24.3%
- Defer Accuracy: 24.7%
- Conflict Rate: 74.7%