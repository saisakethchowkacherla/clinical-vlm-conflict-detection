from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    # Flip probabilities per cell for mock models; ignored by real models.
    mock_flip_rate_conflict: float = 0.218   # matches 27B NF_conflict result
    mock_flip_rate_control: float = 0.006    # matches 27B NF_control result
    mock_cell_a_flip_rate: float = 0.574     # matches 27B cell A result
    mock_conflict_usage_rate: float = 0.0    # model never uses CONFLICT (validated)
    random_seed: int = 20260604


class VLMInferenceModel(ABC):
    """Contract every model adapter must satisfy.

    A model receives a manifest row (with image_path, lab_value, lab_class,
    finding_label, cell) and returns raw text for each condition.
    The scorer (Owner 2) handles parsing — models must NOT pre-parse.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier, e.g. 'medgemma-27b'."""

    @abstractmethod
    def predict_row(
        self,
        image_path: str,
        lab_value: float,
        lab_ref_lower: float,
        lab_ref_upper: float,
        lab_class: str,
    ) -> dict[str, str]:
        """Return raw text for each condition.

        Keys: 'cond0', 'cond1', 'condA'.
        Values: raw model output string (e.g. 'PRESENT', 'ABSENT', 'CONFLICT').
        """

    def run_manifest(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """Run inference over a full manifest and return a predictions DataFrame.

        Output columns: subject_id, study_id, cond0, cond1, condA.
        """
        required = {"subject_id", "study_id", "lab_value", "lab_ref_lower", "lab_ref_upper", "lab_class"}
        missing = required - set(manifest.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {', '.join(sorted(missing))}")

        rows: list[dict[str, object]] = []
        image_col = "image_path" if "image_path" in manifest.columns else None
        for record in manifest.to_dict("records"):
            image_path = str(record[image_col]) if image_col else ""
            preds = self.predict_row(
                image_path=image_path,
                lab_value=float(record["lab_value"]),
                lab_ref_lower=float(record["lab_ref_lower"]),
                lab_ref_upper=float(record["lab_ref_upper"]),
                lab_class=str(record["lab_class"]),
            )
            rows.append(
                {
                    "subject_id": record["subject_id"],
                    "study_id": record["study_id"],
                    "cond0": preds.get("cond0", ""),
                    "cond1": preds.get("cond1", ""),
                    "condA": preds.get("condA", ""),
                }
            )
        return pd.DataFrame(rows)
