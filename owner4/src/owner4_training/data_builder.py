from __future__ import annotations

"""LoRA training data builder (EXPERIMENT_SPEC §8, Owner 4).

Converts a manifest (from the 29,407-study training pool) into supervised
fine-tuning examples for a conflict-aware LoRA.

Two example types:
  faithful  — model should answer based on the image, ignoring a contradicting lab.
              label = finding_label (pos→PRESENT, neg→ABSENT) regardless of lab.
  defer     — ambiguous study (finding uncertain or labellers disagree): label = CONFLICT.

The naive-consistency-LoRA baseline uses the same data but labels every
cond1 as "agree with cond0" — i.e. it learns to be consistent, which may
go text-blind (the 'Consistent but Dangerous' trap).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = "You are a careful radiologist. Answer the question precisely in the exact format requested."
BINARY_QUESTION = (
    "Based on this chest X-ray, is there radiographic evidence of pneumonia? "
    "Answer exactly PRESENT or ABSENT."
)
ABSTENTION_QUESTION = (
    "Based on this chest X-ray, is there radiographic evidence of pneumonia? "
    "If the image and the lab value point in different directions, you may answer CONFLICT. "
    "Answer exactly PRESENT, ABSENT, or CONFLICT."
)


def _lab_statement(lab_value: float, lab_ref_lower: float, lab_ref_upper: float, lab_class: str) -> str:
    qual_map = {"high": "which is elevated", "low": "which is low", "normal": "within normal limits"}
    qual = qual_map.get(lab_class, "within normal limits")
    return (
        f"The patient's white blood cell count is {lab_value:.1f} K/uL "
        f"(reference {lab_ref_lower:.1f}-{lab_ref_upper:.1f}), {qual}."
    )


@dataclass
class TrainingExample:
    subject_id: int | str
    study_id: int | str
    example_type: str          # 'faithful' | 'defer' | 'naive_consistency'
    system_prompt: str
    user_prompt: str
    label: str                 # 'PRESENT' | 'ABSENT' | 'CONFLICT'
    cell: str
    finding_label: str


def build_faithful_example(record: dict) -> TrainingExample:
    lab_stmt = _lab_statement(
        record["lab_value"], record["lab_ref_lower"], record["lab_ref_upper"], record["lab_class"]
    )
    label = "PRESENT" if record["finding_label"] == "pos" else "ABSENT"
    return TrainingExample(
        subject_id=record["subject_id"],
        study_id=record["study_id"],
        example_type="faithful",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"{lab_stmt} {ABSTENTION_QUESTION}",
        label=label,
        cell=record["cell"],
        finding_label=record["finding_label"],
    )


def build_defer_example(record: dict) -> TrainingExample:
    """Uncertain studies: teach the model to output CONFLICT."""
    lab_stmt = _lab_statement(
        record["lab_value"], record["lab_ref_lower"], record["lab_ref_upper"], record["lab_class"]
    )
    return TrainingExample(
        subject_id=record["subject_id"],
        study_id=record["study_id"],
        example_type="defer",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"{lab_stmt} {ABSTENTION_QUESTION}",
        label="CONFLICT",
        cell=record.get("cell", ""),
        finding_label=record.get("finding_label", "uncertain"),
    )


def build_naive_consistency_example(record: dict) -> TrainingExample:
    """Naive baseline: always agree with the image-only prediction regardless of lab."""
    lab_stmt = _lab_statement(
        record["lab_value"], record["lab_ref_lower"], record["lab_ref_upper"], record["lab_class"]
    )
    # label = image finding (learns to be consistent, not necessarily correct)
    label = "PRESENT" if record["finding_label"] == "pos" else "ABSENT"
    return TrainingExample(
        subject_id=record["subject_id"],
        study_id=record["study_id"],
        example_type="naive_consistency",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=f"{lab_stmt} {BINARY_QUESTION}",
        label=label,
        cell=record["cell"],
        finding_label=record["finding_label"],
    )


def build_training_data(
    train_manifest: pd.DataFrame,
    defer_manifest: pd.DataFrame | None = None,
    include_naive: bool = True,
) -> pd.DataFrame:
    """Build a unified training DataFrame from manifests.

    Args:
        train_manifest: The 29,407-study pool with confirmed labels (conflict + control cells).
        defer_manifest: Optional manifest of uncertain studies (no confident label).
        include_naive: If True, also emit naive_consistency examples for the baseline.

    Returns:
        DataFrame of TrainingExample records as rows.
    """
    required = {"subject_id", "study_id", "finding_label", "cell", "lab_value", "lab_ref_lower", "lab_ref_upper", "lab_class"}
    missing = required - set(train_manifest.columns)
    if missing:
        raise ValueError(f"Train manifest missing columns: {', '.join(sorted(missing))}")

    examples: list[TrainingExample] = []

    for record in train_manifest.to_dict("records"):
        examples.append(build_faithful_example(record))
        if include_naive:
            examples.append(build_naive_consistency_example(record))

    if defer_manifest is not None and not defer_manifest.empty:
        for record in defer_manifest.to_dict("records"):
            examples.append(build_defer_example(record))

    rows = [
        {
            "subject_id": ex.subject_id,
            "study_id": ex.study_id,
            "example_type": ex.example_type,
            "system_prompt": ex.system_prompt,
            "user_prompt": ex.user_prompt,
            "label": ex.label,
            "cell": ex.cell,
            "finding_label": ex.finding_label,
        }
        for ex in examples
    ]
    return pd.DataFrame(rows)


def leakage_check(train_df: pd.DataFrame, test_manifest: pd.DataFrame) -> dict[str, object]:
    """Verify no subject_id overlap between training data and test manifest."""
    train_subjects = set(train_df["subject_id"].astype(str))
    test_subjects = set(test_manifest["subject_id"].astype(str))
    overlap = train_subjects & test_subjects
    return {
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
        "overlap_count": len(overlap),
        "leakage_free": len(overlap) == 0,
        "overlap_subjects": sorted(overlap)[:20],  # first 20 for inspection
    }


def write_training_data(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
