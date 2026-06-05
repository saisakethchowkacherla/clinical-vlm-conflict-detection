from __future__ import annotations

from pathlib import Path

import pandas as pd


def apply_mock_training_fix(predictions: pd.DataFrame, strategy: str) -> pd.DataFrame:
    required = {"subject_id", "study_id", "cond0", "cond1"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing column(s): {', '.join(sorted(missing))}")

    fixed = predictions.copy()
    if strategy == "conflict-aware-lora-mock":
        disagreement = fixed["cond0"].astype(str).str.upper() != fixed["cond1"].astype(str).str.upper()
        fixed.loc[disagreement, "cond1"] = fixed.loc[disagreement, "cond0"]
        fixed.loc[disagreement, "condA"] = "CONFLICT"
        suffix = "+conflict-aware-lora-mock"
    elif strategy == "naive-consistency-lora-mock":
        fixed["cond1"] = fixed["cond0"]
        fixed["condA"] = fixed["cond0"]
        suffix = "+naive-consistency-lora-mock"
    else:
        raise ValueError(
            "Unknown strategy. Use conflict-aware-lora-mock or naive-consistency-lora-mock."
        )

    if "model" in fixed.columns:
        fixed["model"] = fixed["model"].astype(str) + suffix
    else:
        fixed["model"] = strategy
    return fixed


def run_mock_fix(predictions_path: Path, strategy: str, output_path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    fixed = apply_mock_training_fix(predictions, strategy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_csv(output_path, index=False)
    return fixed
