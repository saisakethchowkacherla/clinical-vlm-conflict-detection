from __future__ import annotations

"""Mock 'trained' model simulating a conflict-aware LoRA (Owner 4).

Two variants:
  MockFixedLoRA         — conflict-aware LoRA: low flip rate on conflict cells,
                          preserves normal text use on control cells.
  MockNaiveConsistencyLoRA — naive baseline: extremely low flip rate everywhere
                             (text-blind; the 'Consistent but Dangerous' trap).

These plug into Owner 3's VLMInferenceModel contract so Owner 5 can slot them
into the fix-ladder and Owner 2 can score them identically.
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "owner3", "src"))

from owner3_models.contract import InferenceConfig, VLMInferenceModel  # type: ignore[import]

# Conflict-aware LoRA: keeps correct image read even under contradicting lab.
# Target: >= 50% relative reduction in NF_conflict vs 27B baseline (21.8% → <= 10.9%).
_FIXED_LORA_FLIP_RATES: dict[str, float] = {
    "A": 0.08,   # was 57.4% — dramatically reduced
    "B": 0.02,   # was 4.5%
    "C": 0.015,  # unchanged — doesn't over-suppress corroborating signal
    "D": 0.000,
}

# Naive consistency LoRA: learns to agree with cond0, ignores text almost entirely.
# This should show ~0 flips but also ~0 text use — the dangerous trap.
_NAIVE_LORA_FLIP_RATES: dict[str, float] = {
    "A": 0.01,
    "B": 0.01,
    "C": 0.01,
    "D": 0.00,
}


class _MockLoRABase(VLMInferenceModel):
    def __init__(self, config: InferenceConfig, flip_rates: dict[str, float]) -> None:
        self._config = config
        self._flip_rates = flip_rates
        self._rng = random.Random(config.random_seed)

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def predict_row(self, image_path, lab_value, lab_ref_lower, lab_ref_upper, lab_class):
        # Without cell context, use a generic conflict rate.
        rate = self._config.mock_flip_rate_conflict
        answer = "PRESENT"
        cond1 = _flip(answer, rate, self._rng)
        return {"cond0": answer, "cond1": cond1, "condA": cond1}

    def predict_row_with_label(self, finding_label: str, cell: str) -> dict[str, str]:
        image_answer = "PRESENT" if finding_label == "pos" else "ABSENT"
        rate = self._flip_rates.get(cell, self._config.mock_flip_rate_conflict)
        cond1 = _flip(image_answer, rate, self._rng)
        # Conflict-aware LoRA should use CONFLICT on ambiguous cases;
        # naive LoRA never does (it learned consistency, not uncertainty).
        condA = self._condA(image_answer, cond1, cell)
        return {"cond0": image_answer, "cond1": cond1, "condA": condA}

    def _condA(self, image_answer: str, cond1: str, cell: str) -> str:
        return cond1

    def run_manifest_with_labels(self, manifest) -> "pd.DataFrame":  # type: ignore[name-defined]
        import pandas as pd

        required = {"subject_id", "study_id", "finding_label", "cell"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

        rows = []
        for record in manifest.to_dict("records"):
            preds = self.predict_row_with_label(str(record["finding_label"]), str(record["cell"]))
            rows.append({
                "subject_id": record["subject_id"],
                "study_id": record["study_id"],
                "cond0": preds["cond0"],
                "cond1": preds["cond1"],
                "condA": preds["condA"],
            })
        return pd.DataFrame(rows)


class MockFixedLoRA(_MockLoRABase):
    """Conflict-aware LoRA: defends the image on conflict, still uses lab on control."""

    def _condA(self, image_answer: str, cond1: str, cell: str) -> str:
        # On conflict cells where cond1 disagrees with cond0, output CONFLICT.
        if cell in {"A", "B"} and cond1 != image_answer:
            return "CONFLICT"
        return cond1


class MockNaiveConsistencyLoRA(_MockLoRABase):
    """Naive consistency LoRA: learned to be consistent but may be text-blind."""
    pass


def _flip(answer: str, rate: float, rng: random.Random) -> str:
    if rng.random() < rate:
        return "ABSENT" if answer == "PRESENT" else "PRESENT"
    return answer


def make_fixed_lora(seed: int = 20260606) -> MockFixedLoRA:
    config = InferenceConfig(
        model_name="mock-fixed-conflict-lora",
        mock_flip_rate_conflict=0.05,
        mock_flip_rate_control=0.006,
        random_seed=seed,
    )
    return MockFixedLoRA(config, _FIXED_LORA_FLIP_RATES)


def make_naive_lora(seed: int = 20260607) -> MockNaiveConsistencyLoRA:
    config = InferenceConfig(
        model_name="mock-naive-consistency-lora",
        mock_flip_rate_conflict=0.01,
        mock_flip_rate_control=0.01,
        random_seed=seed,
    )
    return MockNaiveConsistencyLoRA(config, _NAIVE_LORA_FLIP_RATES)
