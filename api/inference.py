"""
Real inference using torchxrayvision DenseNet pretrained on chest X-rays.

cond0 — image-only: composite score thresholded at 0.5
cond1 — multimodal: same score, threshold adjusted by lab context
           high WBC  → 0.35  (lab suggests infection)
           low WBC   → 0.65  (lab argues against infection)
           normal WBC→ 0.50

Image classification (COVID / Normal / Viral Pneumonia / Lung Opacity) uses
a dedicated trained head (owner4/outputs_covid/covid_4way_head.pt, 89.1%
test accuracy) when available, falling back to the hand-tuned heuristic
formula (classify_image_type_heuristic) if the trained weights are missing.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TF
import torchvision.transforms as T
import torchxrayvision as xrv
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

_model: xrv.models.DenseNet | None = None

COVID_HEAD_CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
COVID_HEAD_DISPLAY = {
    "COVID": "COVID",
    "Lung_Opacity": "Lung Opacity",
    "Normal": "Normal",
    "Viral Pneumonia": "Viral Pneumonia",
}
COVID_HEAD_WEIGHTS = Path(__file__).parent.parent / "owner4" / "outputs_covid" / "covid_4way_head.pt"

_covid_head: nn.Linear | None = None
_covid_head_loaded = False


def _get_model() -> xrv.models.DenseNet:
    global _model
    if _model is None:
        _model = xrv.models.DenseNet(weights="densenet121-res224-all")
        _model.eval()
    return _model


def _get_covid_head() -> nn.Linear | None:
    """Load the trained 4-way classifier head once. Returns None if weights are missing."""
    global _covid_head, _covid_head_loaded
    if not _covid_head_loaded:
        _covid_head_loaded = True
        if COVID_HEAD_WEIGHTS.exists():
            head = nn.Linear(1024, len(COVID_HEAD_CLASSES))
            head.load_state_dict(torch.load(COVID_HEAD_WEIGHTS, map_location="cpu"))
            head.eval()
            _covid_head = head
    return _covid_head


def _preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
    transform = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    arr = transform(arr)
    return torch.from_numpy(arr).unsqueeze(0)


def _run_model(tensor: torch.Tensor) -> dict[str, float]:
    """Run DenseNet on the preprocessed tensor; return all pathology scores."""
    model = _get_model()
    with torch.no_grad():
        output = model(tensor)[0]
    return {p: round(float(output[i].item()), 4) for i, p in enumerate(model.pathologies) if p}


def _extract_features(tensor: torch.Tensor) -> torch.Tensor:
    """Run only the frozen DenseNet feature extractor (no classifier head)."""
    model = _get_model()
    with torch.no_grad():
        feat = model.features(tensor)
        feat = TF.relu(feat, inplace=True)
        feat = TF.adaptive_avg_pool2d(feat, (1, 1))
        feat = feat.view(feat.size(0), -1)
    return feat


def classify_image_type_trained(tensor: torch.Tensor) -> tuple[str, dict[str, float]]:
    """Classify using the trained 4-way head (89.1% test accuracy)."""
    head = _get_covid_head()
    features = _extract_features(tensor)
    with torch.no_grad():
        logits = head(features)
        probs = TF.softmax(logits, dim=1)[0]
    confidences = {
        COVID_HEAD_DISPLAY[c]: round(float(probs[i].item()), 3)
        for i, c in enumerate(COVID_HEAD_CLASSES)
    }
    predicted_raw = COVID_HEAD_CLASSES[int(probs.argmax().item())]
    return COVID_HEAD_DISPLAY[predicted_raw], confidences


def _composite(scores: dict[str, float]) -> float:
    """Lung Opacity (60%) + Consolidation (25%) + Pneumonia (15%)."""
    return round(
        0.60 * scores.get("Lung Opacity", 0.0)
        + 0.25 * scores.get("Consolidation", 0.0)
        + 0.15 * scores.get("Pneumonia", 0.0),
        4,
    )


def classify_image_type_heuristic(scores: dict[str, float]) -> tuple[str, dict[str, float]]:
    """
    Fallback heuristic — used only if the trained head's weights are missing.

    Classify the X-ray into one of four COVID dataset categories.

    Key discriminators observed from DenseNet scores:
      - Normal         : low Lung Opacity + low Consolidation
      - Viral Pneumonia: high Lung Opacity + high Infiltration + very low Pneumonia score
      - Lung Opacity   : high Pneumonia model score (strongest discriminator)
      - COVID          : moderate-high Lung Opacity + Consolidation + intermediate Pneumonia

    The Pneumonia model score is the primary split between Viral Pneumonia and the rest:
    Viral Pneumonia averages 0.015 vs COVID 0.157 vs Lung Opacity 0.457.
    When Pneumonia > 0.08 the Viral Pneumonia score is penalised accordingly.
    """
    lo      = scores.get("Lung Opacity", 0.0)
    consol  = scores.get("Consolidation", 0.0)
    pneumo  = scores.get("Pneumonia", 0.0)
    infiltr = scores.get("Infiltration", 0.0)

    normal_s = (1 - lo) * 0.50 + (1 - consol) * 0.30 + (1 - pneumo) * 0.20
    covid_s  = lo * 0.25 + consol * 0.35 + min(pneumo, 0.50) * 0.30 + infiltr * 0.10
    viral_s  = lo * 0.45 + infiltr * 0.35 + (1 - pneumo) * 0.20
    lo_s     = pneumo * 0.55 + lo * 0.30 + consol * 0.15

    # Penalise Viral Pneumonia when Pneumonia score is elevated
    if pneumo > 0.08:
        viral_s *= 0.50

    raw = {"Normal": normal_s, "COVID": covid_s, "Viral Pneumonia": viral_s, "Lung Opacity": lo_s}
    total = sum(raw.values()) or 1.0
    confidences = {k: round(v / total, 3) for k, v in raw.items()}
    predicted = max(raw, key=raw.get)
    return predicted, confidences


def _threshold_for(status: str) -> float:
    return {"high": 0.35, "low": 0.65}.get(status, 0.50)


def _predict(score: float, threshold: float) -> str:
    return "PRESENT" if score >= threshold else "ABSENT"


@dataclass
class InferenceResult:
    image_only_prediction: str
    multimodal_prediction: str
    conflict_detected: bool
    defer_gate: str
    lab_value: float
    lab_status: str
    composite_score: float
    image_only_threshold: float
    multimodal_threshold: float
    model_scores: dict[str, float]
    image_type: str
    image_type_confidence: dict[str, float]


def run_inference(
    image_bytes: bytes,
    lab_value: float,
    ref_low: float,
    ref_high: float,
    mime: str = "image/png",
) -> InferenceResult:
    """Single model pass — returns everything needed for the API response."""
    tensor = _preprocess(image_bytes)
    scores = _run_model(tensor)
    composite = _composite(scores)

    if _get_covid_head() is not None:
        image_type, image_type_confidence = classify_image_type_trained(tensor)
    else:
        image_type, image_type_confidence = classify_image_type_heuristic(scores)

    status = lab_status(lab_value, ref_low, ref_high)
    t_image = 0.50
    t_multi = _threshold_for(status)

    img_pred = _predict(composite, t_image)
    mm_pred  = _predict(composite, t_multi)
    conflict = img_pred != mm_pred

    return InferenceResult(
        image_only_prediction=img_pred,
        multimodal_prediction=mm_pred,
        conflict_detected=conflict,
        defer_gate="DEFER" if conflict else "PROCEED",
        lab_value=lab_value,
        lab_status=status,
        composite_score=composite,
        image_only_threshold=t_image,
        multimodal_threshold=t_multi,
        model_scores=scores,
        image_type=image_type,
        image_type_confidence=image_type_confidence,
    )


def lab_status(lab_value: float, ref_low: float, ref_high: float) -> str:
    if lab_value < ref_low:
        return "low"
    if lab_value > ref_high:
        return "high"
    return "normal"
