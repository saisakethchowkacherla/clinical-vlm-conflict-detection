from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class MockModelProfile:
    name: str
    flip_cells: frozenset[str]
    degenerate_absent: bool = False
    conflict_when_flipped: bool = False


PROFILES = {
    "mock-medgemma-4b": MockModelProfile(
        name="mock-medgemma-4b",
        flip_cells=frozenset({"A"}),
        conflict_when_flipped=False,
    ),
    "mock-medgemma-27b": MockModelProfile(
        name="mock-medgemma-27b",
        flip_cells=frozenset({"A", "B"}),
        conflict_when_flipped=False,
    ),
    "mock-qwen-degenerate": MockModelProfile(
        name="mock-qwen-degenerate",
        flip_cells=frozenset(),
        degenerate_absent=True,
        conflict_when_flipped=False,
    ),
    "mock-fixed-model": MockModelProfile(
        name="mock-fixed-model",
        flip_cells=frozenset({"A"}),
        conflict_when_flipped=True,
    ),
}


def label_to_answer(finding_label: str) -> str:
    if finding_label == "pos":
        return "PRESENT"
    if finding_label == "neg":
        return "ABSENT"
    return "ABSENT"


def opposite(answer: str) -> str:
    return "ABSENT" if answer == "PRESENT" else "PRESENT"


def generate_mock_predictions(manifest: pd.DataFrame, profile: MockModelProfile) -> pd.DataFrame:
    required = {"subject_id", "study_id", "finding_label", "cell"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing column(s): {', '.join(sorted(missing))}")

    rows: list[dict[str, object]] = []
    for row in manifest.to_dict("records"):
        correct = label_to_answer(row["finding_label"])
        cond0 = "ABSENT" if profile.degenerate_absent else correct
        should_flip = row["cell"] in profile.flip_cells and not profile.degenerate_absent
        cond1 = opposite(correct) if should_flip else correct
        cond2 = cond1
        condA = "CONFLICT" if should_flip and profile.conflict_when_flipped else cond1
        rows.append(
            {
                "subject_id": row["subject_id"],
                "study_id": row["study_id"],
                "model": profile.name,
                "cond0": cond0,
                "cond1": cond1,
                "cond2": cond2,
                "condA": condA,
            }
        )
    return pd.DataFrame(rows)


def run_mock_inference(manifest_path: Path, model: str, output_path: Path) -> pd.DataFrame:
    if model not in PROFILES:
        raise ValueError(f"Unknown mock model {model!r}. Options: {', '.join(PROFILES)}")
    manifest = pd.read_csv(manifest_path)
    predictions = generate_mock_predictions(manifest, PROFILES[model])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions
