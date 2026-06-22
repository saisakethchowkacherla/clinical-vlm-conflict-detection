"""
Real inference using torchxrayvision DenseNet121 pretrained on chest X-rays.

Produces the same output CSV format as mock_inference so all downstream
owners (2, 4, 5, 6) work without any changes.

Columns produced: subject_id, study_id, model, cond0, cond1, cond2, condA

  cond0  — image-only prediction, threshold 0.5
  cond1  — multimodal prediction, threshold shifted by lab_class:
             high WBC  → threshold 0.35  (lab suggests infection)
             low WBC   → threshold 0.65  (lab argues against infection)
             normal WBC→ threshold 0.50
  cond2  — same as cond1 (reserved)
  condA  — CONFLICT if cond0 != cond1, else cond1
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchxrayvision as xrv
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_NAME = "densenet121-res224-all"
THRESHOLD_DEFAULT = 0.50
THRESHOLD_HIGH_WBC = 0.35
THRESHOLD_LOW_WBC = 0.65


def _load_model() -> xrv.models.DenseNet:
    model = xrv.models.DenseNet(weights=MODEL_NAME)
    model.eval()
    return model


def _composite_score(output: torch.Tensor, pathologies: list[str]) -> float:
    """Lung Opacity (60%) + Consolidation (25%) + Pneumonia (15%)."""
    idx = {p: i for i, p in enumerate(pathologies) if p}
    score = (
        0.60 * float(output[idx["Lung Opacity"]].item())
        + 0.25 * float(output[idx["Consolidation"]].item())
        + 0.15 * float(output[idx["Pneumonia"]].item())
    )
    return score


def _infer_image(model: xrv.models.DenseNet, image_path: str) -> float:
    """Return composite pathology score for one image."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
    transform = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    arr = transform(arr)
    tensor = torch.from_numpy(arr).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)[0]
    return _composite_score(output, list(model.pathologies))


def _predict(score: float, threshold: float) -> str:
    return "PRESENT" if score >= threshold else "ABSENT"


def _threshold_for_lab(lab_class: str) -> float:
    if lab_class == "high":
        return THRESHOLD_HIGH_WBC
    if lab_class == "low":
        return THRESHOLD_LOW_WBC
    return THRESHOLD_DEFAULT


def run_real_inference(manifest_path: Path, output_path: Path) -> pd.DataFrame:
    required = {"subject_id", "study_id", "image_path", "lab_class"}
    manifest = pd.read_csv(manifest_path)
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing column(s): {', '.join(sorted(missing))}")

    model = _load_model()
    rows: list[dict[str, object]] = []

    for i, row in enumerate(manifest.to_dict("records"), 1):
        image_path = str(row["image_path"])
        if not Path(image_path).exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}\n"
                "Re-run the Owner 1 pipeline to rebuild the manifest with real image paths."
            )

        score = _infer_image(model, image_path)
        cond0 = _predict(score, THRESHOLD_DEFAULT)
        cond1 = _predict(score, _threshold_for_lab(str(row["lab_class"])))
        cond2 = cond1
        condA = "CONFLICT" if cond0 != cond1 else cond1

        rows.append({
            "subject_id": row["subject_id"],
            "study_id":   row["study_id"],
            "model":      "densenet121-xrv",
            "cond0":      cond0,
            "cond1":      cond1,
            "cond2":      cond2,
            "condA":      condA,
        })

        if i % 50 == 0:
            print(f"  {i}/{len(manifest)} images processed")

    predictions = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions
