# Benchmark Taxonomy

## Cell Types

The first benchmark uses four pneumonia/WBC cells:

| Cell | Image Finding | Lab Signal | Relationship |
| --- | --- | --- | --- |
| A | Pneumonia present | WBC normal or low | Contradiction |
| B | Pneumonia absent | WBC high | Contradiction |
| C | Pneumonia present | WBC high | Corroboration |
| D | Pneumonia absent | WBC normal or low | Corroboration |

## Broader Taxonomy

- **Contradiction:** image and structured signal point in opposite directions.
- **Corroboration:** image and structured signal point in the same direction.
- **Complement:** image alone is insufficient, and the structured signal legitimately helps resolve the answer.
- **Irrelevance:** structured signal is present but should not affect the image-grounded answer.

## Required Guardrails

- Do not claim a conflict effect for any model that fails the baseline-validity gate.
- Always compare conflict cells against corroboration controls.
- Always report patient-disjoint leakage checks from Owner 1.
- Do not treat cross-labeler disagreement as image ground truth.
- Use deterministic parsing and scoring, not an LLM judge.
