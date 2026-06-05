from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd


ANSWER_PATTERN = re.compile(r"\b(PRESENT|ABSENT)\b")


def parse_answer(output: object) -> str:
    if output is None or (isinstance(output, float) and math.isnan(output)):
        return "UNPARSED"
    text = str(output).upper()
    if "CONFLICT" in text:
        return "CONFLICT"
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1]
    return "UNPARSED"


def apply_defer_gate(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "study_id", "cond0", "cond1"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing column(s): {', '.join(sorted(missing))}")

    gated = predictions.copy()
    cond0 = gated["cond0"].map(parse_answer)
    cond1 = gated["cond1"].map(parse_answer)
    disagreement = cond0.isin({"PRESENT", "ABSENT"}) & cond1.isin({"PRESENT", "ABSENT"}) & (cond0 != cond1)
    gated.loc[disagreement, "cond1"] = "CONFLICT"
    gated.loc[disagreement, "condA"] = "CONFLICT"
    if "model" in gated.columns:
        gated["model"] = gated["model"].astype(str) + "+defer-gate"
    else:
        gated["model"] = "defer-gate"
    return gated


def run_gate(predictions_path: Path, strategy: str, output_path: Path) -> pd.DataFrame:
    if strategy != "defer-on-disagree":
        raise ValueError("Only defer-on-disagree is implemented.")
    predictions = pd.read_csv(predictions_path)
    gated = apply_defer_gate(predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gated.to_csv(output_path, index=False)
    return gated
