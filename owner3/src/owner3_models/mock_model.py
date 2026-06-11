from __future__ import annotations

import random

from .contract import InferenceConfig, VLMInferenceModel

# Cell A flip rate in the validated 27B result is 57.4%; cell B is 4.5%.
# The mock reproduces these asymmetric rates to mirror real behaviour.
_CELL_FLIP_RATES: dict[str, float] = {
    "A": 0.574,
    "B": 0.045,
    "C": 0.015,
    "D": 0.000,
}


class MockVLMModel(VLMInferenceModel):
    """Deterministic mock that reproduces the validated MedGemma-27B flip profile.

    cond0 answers correctly based on the finding_label passed via lab_class context.
    cond1 flips the answer with the per-cell probability from the validated kernel.
    condA never uses CONFLICT (0/480 validated result).

    Because predict_row does not receive the cell or finding_label directly, the
    caller (run_manifest_with_labels) should be used for cell-aware mocking.
    Use predict_row only when cell is embedded in lab_class as 'cell:<X>:<label>'.
    """

    def __init__(self, config: InferenceConfig) -> None:
        self._config = config
        self._rng = random.Random(config.random_seed)

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def predict_row(
        self,
        image_path: str,
        lab_value: float,
        lab_ref_lower: float,
        lab_ref_upper: float,
        lab_class: str,
    ) -> dict[str, str]:
        """Basic prediction; uses generic flip rates without cell knowledge."""
        image_answer = "PRESENT"
        flip_rate = self._config.mock_flip_rate_conflict
        cond1_answer = _flip(image_answer, flip_rate, self._rng)
        condA_answer = cond1_answer  # never uses CONFLICT (validated)
        return {"cond0": image_answer, "cond1": cond1_answer, "condA": condA_answer}

    def predict_row_with_label(
        self,
        finding_label: str,
        cell: str,
    ) -> dict[str, str]:
        """Cell-aware prediction used by run_manifest_with_labels."""
        image_answer = "PRESENT" if finding_label == "pos" else "ABSENT"
        flip_rate = _CELL_FLIP_RATES.get(cell, self._config.mock_flip_rate_conflict)
        cond1_answer = _flip(image_answer, flip_rate, self._rng)
        # condA: model never flags CONFLICT (0/480); just repeats cond1
        condA_answer = cond1_answer
        return {"cond0": image_answer, "cond1": cond1_answer, "condA": condA_answer}

    def run_manifest_with_labels(self, manifest) -> "pd.DataFrame":  # type: ignore[name-defined]
        import pandas as pd

        required = {"subject_id", "study_id", "finding_label", "cell"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

        rows: list[dict[str, object]] = []
        for record in manifest.to_dict("records"):
            preds = self.predict_row_with_label(
                finding_label=str(record["finding_label"]),
                cell=str(record["cell"]),
            )
            rows.append(
                {
                    "subject_id": record["subject_id"],
                    "study_id": record["study_id"],
                    "cond0": preds["cond0"],
                    "cond1": preds["cond1"],
                    "condA": preds["condA"],
                }
            )
        return pd.DataFrame(rows)


def _flip(answer: str, rate: float, rng: random.Random) -> str:
    if rng.random() < rate:
        return "ABSENT" if answer == "PRESENT" else "PRESENT"
    return answer


def make_mock_model(
    model_name: str = "mock-medgemma-27b",
    flip_rate_conflict: float = 0.218,
    flip_rate_control: float = 0.006,
    seed: int = 20260604,
) -> MockVLMModel:
    config = InferenceConfig(
        model_name=model_name,
        mock_flip_rate_conflict=flip_rate_conflict,
        mock_flip_rate_control=flip_rate_control,
        random_seed=seed,
    )
    return MockVLMModel(config)
