from __future__ import annotations

"""Baseline-validity gate (EXPERIMENT_SPEC §4).

Run on cond0 predictions before any conflict claim.
A model is VALID only if sensitivity >= 0.15, specificity >= 0.15,
and accuracy exceeds majority baseline by >= 0.03.
"""

import pandas as pd


def run_baseline_gate(predictions: pd.DataFrame, manifest: pd.DataFrame) -> dict[str, object]:
    """Compute baseline-validity gate for a model's cond0 predictions.

    Args:
        predictions: DataFrame with subject_id, study_id, cond0.
        manifest: DataFrame with subject_id, study_id, finding_label, cell.

    Returns:
        Dict with gate result and diagnostic metrics.
    """
    merged = manifest.merge(predictions[["subject_id", "study_id", "cond0"]], on=["subject_id", "study_id"], how="inner")
    if merged.empty:
        return {"baseline_valid": False, "reason": "no overlapping rows", "n": 0}

    pos = merged["finding_label"] == "pos"
    neg = merged["finding_label"] == "neg"
    pred_present = merged["cond0"].str.upper().str.contains("PRESENT", na=False)
    pred_absent = merged["cond0"].str.upper().str.contains("ABSENT", na=False)

    tp = int((pred_present & pos).sum())
    tn = int((pred_absent & neg).sum())
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    n = len(merged)

    sensitivity = tp / n_pos if n_pos > 0 else 0.0
    specificity = tn / n_neg if n_neg > 0 else 0.0
    accuracy = (tp + tn) / n if n > 0 else 0.0
    majority = max(n_pos, n_neg) / n if n > 0 else 0.0

    valid = sensitivity >= 0.15 and specificity >= 0.15 and (accuracy - majority) >= 0.03

    reasons: list[str] = []
    if sensitivity < 0.15:
        reasons.append(f"sensitivity {sensitivity:.3f} < 0.15")
    if specificity < 0.15:
        reasons.append(f"specificity {specificity:.3f} < 0.15")
    if (accuracy - majority) < 0.03:
        reasons.append(f"accuracy-majority {accuracy - majority:.3f} < 0.03")

    return {
        "baseline_valid": bool(valid),
        "reason": "PASS" if valid else "; ".join(reasons),
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "accuracy": round(accuracy, 4),
        "majority_baseline": round(majority, 4),
        "accuracy_minus_majority": round(accuracy - majority, 4),
    }


def gate_table(model_results: list[dict[str, object]]) -> pd.DataFrame:
    """Build a summary gate table from a list of per-model gate dicts."""
    return pd.DataFrame(model_results)
