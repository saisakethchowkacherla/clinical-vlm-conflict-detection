# Inference Contract

## Input

Use Owner 1 manifest rows. Each row should include:

- `subject_id`
- `study_id`
- `dicom_id`
- `image_path`
- `finding_label`
- `lab_value`
- `lab_ref_lower`
- `lab_ref_upper`
- `lab_class`
- `cell`

## Conditions

Generate one model output per study for each condition:

- `cond0`: image only.
- `cond1`: image + plain WBC statement.
- `cond2`: image + WBC statement + clinical-lean sentence.
- `condA`: image + WBC statement + option to answer `CONFLICT`.

## Output CSV

Save prediction files in this shape:

```csv
subject_id,study_id,model,cond0,cond1,cond2,condA
10000001,5001,google/medgemma-4b-it,PRESENT,ABSENT,ABSENT,CONFLICT
```

Owner 2 needs at minimum:

- `subject_id`
- `study_id`
- `cond0`
- `cond1`

## Generation Rules

- Use greedy decoding.
- Do not sample.
- Use `max_new_tokens=16`.
- Keep the system prompt fixed.
- Preserve raw text output; do not manually clean it before scoring.
