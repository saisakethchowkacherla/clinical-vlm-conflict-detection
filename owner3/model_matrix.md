# Model Matrix

## Required Starting Models

| Model | Family | Expected Role | Notes |
| --- | --- | --- | --- |
| `google/medgemma-4b-it` | Medical VLM | Kernel reproduction | Should pass baseline gate |
| `google/medgemma-27b-it` | Medical VLM | Main reference model | Day-1 reproduction target |
| `Qwen/Qwen2.5-VL-7B-Instruct` | General VLM | Gate comparison | Expected to fail or be weak on CXR pneumonia |

## Expansion Candidates

| Model | Reason |
| --- | --- |
| CheXagent-8B | Medical CXR-focused comparison |
| MAIRA-2 | Medical imaging comparison |
| InternVL variants | General VLM family comparison |
| Lingshu / HuatuoGPT-Vision | Medical-tuned open models |

## Gate Rule

Do not run or report conflict claims for a model unless it passes the image-only baseline-validity gate:

- sensitivity >= 0.15;
- specificity >= 0.15;
- accuracy is at least 0.03 above majority baseline.

Failing the gate is still a useful result; it means the model is inconclusive for this benchmark.
