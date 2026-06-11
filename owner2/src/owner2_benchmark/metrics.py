from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd


ANSWER_PATTERN = re.compile(r"\b(PRESENT|ABSENT)\b")


def parse_answer(output: object) -> str:
    """Deterministically parse model text into present/absent/conflict/unparsed."""
    if output is None or (isinstance(output, float) and math.isnan(output)):
        return "unparsed"
    text = str(output).upper()
    if "CONFLICT" in text:
        return "conflict"
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].lower()
    return "unparsed"


def is_correct(prediction: str, finding_label: str) -> bool:
    if prediction not in {"present", "absent"}:
        return False
    return (prediction == "present") == (finding_label == "pos")


def safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def rate_record(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "rate": safe_rate(numerator, denominator),
    }


def baseline_metrics(scored: pd.DataFrame) -> dict[str, object]:
    pos = scored["finding_label"] == "pos"
    neg = scored["finding_label"] == "neg"
    correct = scored["cond0_correct"]
    sensitivity_num = int(((scored["cond0_parsed"] == "present") & pos).sum())
    specificity_num = int(((scored["cond0_parsed"] == "absent") & neg).sum())
    accuracy_num = int(correct.sum())
    majority = max(int(pos.sum()), int(neg.sum())) / len(scored) if len(scored) else 0
    accuracy = accuracy_num / len(scored) if len(scored) else 0
    sensitivity = safe_rate(sensitivity_num, int(pos.sum()))
    specificity = safe_rate(specificity_num, int(neg.sum()))
    valid = (
        sensitivity is not None
        and specificity is not None
        and sensitivity >= 0.15
        and specificity >= 0.15
        and (accuracy - majority) >= 0.03
    )
    return {
        "n": int(len(scored)),
        "accuracy": rate_record(accuracy_num, len(scored)),
        "sensitivity": rate_record(sensitivity_num, int(pos.sum())),
        "specificity": rate_record(specificity_num, int(neg.sum())),
        "majority_baseline": majority,
        "accuracy_minus_majority": accuracy - majority,
        "baseline_valid": bool(valid),
    }


def negative_flip(scored: pd.DataFrame, cells: set[str]) -> dict[str, object]:
    subset = scored.loc[scored["cell"].isin(cells) & scored["cond0_correct"]].copy()
    binary_cond1 = subset["cond1_parsed"].isin({"present", "absent"})
    flips = binary_cond1 & (~subset["cond1_correct"])
    return rate_record(int(flips.sum()), int(len(subset)))


def a_vs_c_contrast(scored: pd.DataFrame) -> dict[str, object]:
    return {
        "cell_A": negative_flip(scored, {"A"}),
        "cell_C": negative_flip(scored, {"C"}),
    }


def abstention_metrics(scored: pd.DataFrame) -> dict[str, object]:
    if "condA_parsed" not in scored.columns:
        return {"available": False}
    conflict_cells = scored["cell"].isin({"A", "B"})
    conflict_usage = scored["condA_parsed"] == "conflict"
    conflict_cell_flips = (
        conflict_cells
        & scored["cond0_correct"]
        & scored["cond1_parsed"].isin({"present", "absent"})
        & (~scored["cond1_correct"])
    )
    return {
        "available": True,
        "conflict_usage_overall": rate_record(int(conflict_usage.sum()), int(len(scored))),
        "rescue_among_conflict_cell_flips": rate_record(
            int((conflict_cell_flips & conflict_usage).sum()), int(conflict_cell_flips.sum())
        ),
    }


def score_predictions(manifest: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    required_manifest = {"subject_id", "study_id", "finding_label", "cell"}
    required_predictions = {"subject_id", "study_id", "cond0", "cond1"}
    missing_manifest = required_manifest - set(manifest.columns)
    missing_predictions = required_predictions - set(predictions.columns)
    if missing_manifest:
        raise ValueError(f"Manifest missing column(s): {', '.join(sorted(missing_manifest))}")
    if missing_predictions:
        raise ValueError(f"Predictions missing column(s): {', '.join(sorted(missing_predictions))}")

    scored = manifest.merge(
        predictions,
        on=["subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
    ).copy()
    if scored.empty:
        raise ValueError("No overlapping subject_id/study_id rows between manifest and predictions.")

    for condition in ["cond0", "cond1", "condA"]:
        if condition in scored.columns:
            scored[f"{condition}_parsed"] = scored[condition].map(parse_answer)
            if condition != "condA":
                scored[f"{condition}_correct"] = [
                    is_correct(prediction, label)
                    for prediction, label in zip(
                        scored[f"{condition}_parsed"], scored["finding_label"], strict=False
                    )
                ]

    nf_conflict = negative_flip(scored, {"A", "B"})
    nf_control = negative_flip(scored, {"C", "D"})
    gap = None
    if nf_conflict["rate"] is not None and nf_control["rate"] is not None:
        gap = nf_conflict["rate"] - nf_control["rate"]

    metrics = {
        "rows_scored": int(len(scored)),
        "baseline": baseline_metrics(scored),
        "negative_flip": {
            "conflict_A_B": nf_conflict,
            "control_C_D": nf_control,
            "gap": gap,
            "a_vs_c": a_vs_c_contrast(scored),
        },
        "abstention": abstention_metrics(scored),
        "parse_counts": {
            column: scored[column].value_counts(dropna=False).to_dict()
            for column in ["cond0_parsed", "cond1_parsed", "condA_parsed"]
            if column in scored.columns
        },
    }
    return scored, metrics


def write_outputs(scored: pd.DataFrame, metrics: dict[str, object], metrics_path: Path) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    scored_path = metrics_path.with_name(metrics_path.stem + "_scored.csv")
    scored.to_csv(scored_path, index=False)
