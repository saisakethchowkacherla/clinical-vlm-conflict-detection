"""
Classifier-head fine-tuning for clinical VLM conflict reduction.

Architecture note — why standard LoRA does not apply here
---------------------------------------------------------
PEFT LoRA decomposes transformer *attention* weight matrices W into
W + A @ B (low-rank update). DenseNet121 has no attention layers; its
feature extractor is entirely convolutional. Its only Linear layer is
the final classifier (nn.Linear 1024 -> 18 pathologies). Applying LoRA
to this single small layer gives no practical benefit over plain
fine-tuning because (a) 1024 * 18 = 18 432 parameters is already a
"low-rank" matrix in practice, and (b) the bottleneck is the frozen
convolutional features, not the linear head.

What we do instead (CNN adapter / head fine-tuning)
-----------------------------------------------------
1. Freeze all convolutional feature layers.
2. Pre-compute 1024-dim feature vectors for every image once
   (expensive but one-time; features are stable under frozen conv).
3. Fine-tune only model.classifier (1024 -> 18) via BCE loss on the
   composite pneumonia score vs the image finding label.
4. Save the adapted weights to disk.
5. Re-run the full inference pipeline (cond0 / cond1 / condA) on the
   held-out test split and score with the Owner 2 benchmark.

Training objective
------------------
composite = 0.60 * Lung_Opacity + 0.25 * Consolidation + 0.15 * Pneumonia

Push this score toward 1.0 for positive images and 0.0 for negative
images, making the model's predictions robust to the WBC-driven
threshold shifts (high WBC -> threshold 0.35, low -> 0.65) that cause
the negative-flip phenomenon.

Usage
-----
  python -m owner4_training.lora_train \\
      --manifest   owner1/outputs_covid/kernel_manifest.csv \\
      --output-dir owner4/outputs_covid \\
      --epochs     5 \\
      --max-train  2000 \\
      --max-test   500

If --max-train is omitted the entire train split is used (slow on CPU).
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchxrayvision as xrv
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

# ── pathology indices for composite score ──────────────────────────────────────
COMPOSITE_WEIGHTS = {
    "Lung Opacity":  0.60,
    "Consolidation": 0.25,
    "Pneumonia":     0.15,
}

THRESHOLD_DEFAULT  = 0.50
THRESHOLD_HIGH_WBC = 0.35
THRESHOLD_LOW_WBC  = 0.65


# ── model helpers ──────────────────────────────────────────────────────────────

def load_base_model() -> xrv.models.DenseNet:
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    return model


def pathology_index(model: xrv.models.DenseNet) -> dict[str, int]:
    return {p: i for i, p in enumerate(model.pathologies) if p}


def composite_from_probs(probs: torch.Tensor, idx: dict[str, int]) -> torch.Tensor:
    """Weighted sum of three pathology probabilities. probs shape: (N, 18)."""
    return sum(w * probs[:, idx[p]] for p, w in COMPOSITE_WEIGHTS.items())  # type: ignore[return-value]


def preprocess_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)
    transform = T.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
    arr = transform(arr)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 1, 224, 224)


# ── feature pre-computation ────────────────────────────────────────────────────

def extract_features_batch(
    model: xrv.models.DenseNet,
    image_paths: list[str],
    verbose: bool = True,
) -> torch.Tensor:
    """
    Run the DenseNet feature extractor (no classifier) on each image once.
    Returns tensor of shape (N, 1024). This is the expensive one-time step.
    """
    import torch.nn.functional as F

    features_list: list[torch.Tensor] = []
    n = len(image_paths)
    for i, path in enumerate(image_paths, 1):
        tensor = preprocess_image(path)
        with torch.no_grad():
            feat = model.features(tensor)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)  # (1, 1024)
        features_list.append(feat)
        if verbose and i % 200 == 0:
            print(f"  feature extraction: {i}/{n} images", flush=True)
    return torch.cat(features_list, dim=0)  # (N, 1024)


# ── training ───────────────────────────────────────────────────────────────────

def train_classifier(
    model: xrv.models.DenseNet,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> None:
    """
    Fine-tune only model.classifier (nn.Linear 1024->18) on pre-computed features.
    All convolutional layers stay frozen throughout.
    """
    idx = pathology_index(model)

    # Freeze feature extractor; only classifier is trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    model.train()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    criterion = nn.BCELoss()

    n = len(train_labels)
    perm = torch.randperm(n)
    train_features = train_features[perm]
    train_labels = train_labels[perm]

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            feats = train_features[start : start + batch_size]
            labels = train_labels[start : start + batch_size]

            logits = model.classifier(feats)          # (B, 18) — raw
            probs  = torch.sigmoid(logits)            # (B, 18) — [0,1]
            comp   = composite_from_probs(probs, idx) # (B,)

            loss = criterion(comp, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        print(f"  epoch {epoch}/{epochs}  avg_loss={epoch_loss/batches:.4f}", flush=True)

    # Restore eval mode; features stay frozen for inference
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


# ── inference (produces Owner 2 format predictions CSV) ───────────────────────

def run_inference(
    model: xrv.models.DenseNet,
    manifest_slice: pd.DataFrame,
    model_name: str = "densenet121-xrv",
) -> pd.DataFrame:
    """
    Run full inference pipeline on a manifest slice.
    Returns predictions in Owner 2 format (subject_id, study_id, cond0, cond1, condA).
    """
    import torch.nn.functional as F

    idx = pathology_index(model)
    rows: list[dict] = []

    for rec in manifest_slice.to_dict("records"):
        image_path = str(rec["image_path"])
        if not Path(image_path).exists():
            continue

        tensor = preprocess_image(image_path)
        with torch.no_grad():
            # Full forward (features + classifier + sigmoid) — same as normal inference
            feat = model.features(tensor)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
            logits = model.classifier(feat)
            probs  = torch.sigmoid(logits)

        comp = composite_from_probs(probs, idx).item()

        lab_class = str(rec.get("lab_class", "normal"))
        t_image = THRESHOLD_DEFAULT
        t_multi = {"high": THRESHOLD_HIGH_WBC, "low": THRESHOLD_LOW_WBC}.get(lab_class, THRESHOLD_DEFAULT)

        cond0 = "PRESENT" if comp >= t_image else "ABSENT"
        cond1 = "PRESENT" if comp >= t_multi else "ABSENT"
        condA = "CONFLICT" if cond0 != cond1 else cond1

        rows.append({
            "subject_id": rec["subject_id"],
            "study_id":   rec["study_id"],
            "model":      model_name,
            "cond0":      cond0,
            "cond1":      cond1,
            "cond2":      cond1,
            "condA":      condA,
        })

    return pd.DataFrame(rows)


# ── Owner 2 metrics (inline so no import path issues) ─────────────────────────

def _parse(output: object) -> str:
    import math, re
    if output is None or (isinstance(output, float) and math.isnan(output)):
        return "unparsed"
    text = str(output).upper()
    if "CONFLICT" in text:
        return "conflict"
    m = re.findall(r"\b(PRESENT|ABSENT)\b", text)
    return m[-1].lower() if m else "unparsed"


def _is_correct(pred: str, label: str) -> bool:
    return pred in {"present", "absent"} and (pred == "present") == (label == "pos")


def quick_metrics(manifest: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Compute NF_conflict, NF_control, gap, accuracy."""
    scored = manifest.merge(predictions, on=["subject_id", "study_id"], how="inner")
    if scored.empty:
        return {}
    scored["c0"] = scored["cond0"].map(_parse)
    scored["c1"] = scored["cond1"].map(_parse)
    scored["c0_ok"] = [_is_correct(p, l) for p, l in zip(scored["c0"], scored["finding_label"])]
    scored["c1_ok"] = [_is_correct(p, l) for p, l in zip(scored["c1"], scored["finding_label"])]

    def nf(cells: set[str]) -> tuple[int, int]:
        sub = scored.loc[scored["cell"].isin(cells) & scored["c0_ok"]]
        binary = sub["c1"].isin({"present", "absent"})
        flips  = binary & (~sub["c1_ok"])
        return int(flips.sum()), int(len(sub))

    nf_conf_n, nf_conf_d = nf({"A", "B"})
    nf_ctrl_n, nf_ctrl_d = nf({"C", "D"})
    acc = int(scored["c0_ok"].sum()) / len(scored)
    gap = (nf_conf_n / nf_conf_d if nf_conf_d else None,
           nf_ctrl_n / nf_ctrl_d if nf_ctrl_d else None)

    return {
        "n_tested":    len(scored),
        "accuracy":    round(acc, 4),
        "NF_conflict": {"n": nf_conf_n, "d": nf_conf_d, "rate": round(nf_conf_n / nf_conf_d, 4) if nf_conf_d else None},
        "NF_control":  {"n": nf_ctrl_n, "d": nf_ctrl_d, "rate": round(nf_ctrl_n / nf_ctrl_d, 4) if nf_ctrl_d else None},
        "gap":         round(gap[0] - gap[1], 4) if None not in gap else None,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classifier-head fine-tuning for conflict reduction.")
    parser.add_argument("--manifest",   required=True, type=Path, help="kernel_manifest.csv path")
    parser.add_argument("--output-dir", default=Path("owner4/outputs_covid"), type=Path)
    parser.add_argument("--epochs",     default=5,    type=int)
    parser.add_argument("--max-train",  default=None, type=int, help="Cap training images (default: all)")
    parser.add_argument("--max-test",   default=None, type=int, help="Cap test images (default: all)")
    parser.add_argument("--lr",         default=1e-3, type=float)
    parser.add_argument("--batch-size", default=64,   type=int)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    print(f"Manifest loaded: {len(manifest):,} rows, cells: "
          f"{manifest['cell'].value_counts().to_dict()}")

    # ── split train / test ────────────────────────────────────────────────────
    train_df = manifest.loc[manifest["split"] == "train"].copy().reset_index(drop=True)
    test_df  = manifest.loc[manifest["split"] == "test"].copy().reset_index(drop=True)

    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(args.max_train, random_state=20260604).reset_index(drop=True)
        print(f"Training capped to {args.max_train} images.")
    if args.max_test and len(test_df) > args.max_test:
        test_df = test_df.sample(args.max_test, random_state=20260604).reset_index(drop=True)
        print(f"Test capped to {args.max_test} images.")

    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

    # ── load model ────────────────────────────────────────────────────────────
    print("\nLoading DenseNet121...")
    model = load_base_model()

    # ── baseline evaluation ───────────────────────────────────────────────────
    print("\n[BASELINE] Running inference on test set before training...")
    t0 = time.time()
    baseline_preds = run_inference(model, test_df, model_name="densenet121-xrv-baseline")
    baseline_metrics = quick_metrics(test_df, baseline_preds)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── pre-compute train features ────────────────────────────────────────────
    print(f"\n[FEATURE EXTRACTION] {len(train_df):,} training images...")
    t0 = time.time()
    train_image_paths = train_df["image_path"].tolist()
    train_features = extract_features_batch(model, train_image_paths, verbose=True)
    train_labels   = torch.tensor(
        (train_df["finding_label"] == "pos").astype(float).values, dtype=torch.float32
    )
    print(f"  done in {time.time()-t0:.1f}s — features shape: {tuple(train_features.shape)}")

    # ── train ─────────────────────────────────────────────────────────────────
    print(f"\n[TRAINING] Fine-tuning classifier head — {args.epochs} epochs, lr={args.lr}...")
    t0 = time.time()
    train_classifier(
        model, train_features, train_labels,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )
    print(f"  training done in {time.time()-t0:.1f}s")

    # ── save adapted weights ──────────────────────────────────────────────────
    weights_path = args.output_dir / "densenet121_head_finetuned.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"  saved: {weights_path}")

    # ── post-training evaluation ──────────────────────────────────────────────
    print("\n[POST-TRAINING] Running inference on test set after training...")
    t0 = time.time()
    tuned_preds = run_inference(model, test_df, model_name="densenet121-xrv-head-finetuned")
    tuned_metrics = quick_metrics(test_df, tuned_preds)
    print(f"  done in {time.time()-t0:.1f}s")

    # ── save prediction CSVs ──────────────────────────────────────────────────
    baseline_preds.to_csv(args.output_dir / "baseline_predictions.csv", index=False)
    tuned_preds.to_csv(   args.output_dir / "finetuned_predictions.csv", index=False)

    # ── comparison table ──────────────────────────────────────────────────────
    def fmt(m: dict, label: str) -> str:
        nfc = m.get("NF_conflict", {})
        nfk = m.get("NF_control",  {})
        return (
            f"  {label:<30}  acc={m.get('accuracy', 'N/A'):.4f}  "
            f"NF_conflict={nfc.get('rate', 'N/A')} ({nfc.get('n','?')}/{nfc.get('d','?')})  "
            f"NF_control={nfk.get('rate', 'N/A')} ({nfk.get('n','?')}/{nfk.get('d','?')})  "
            f"gap={m.get('gap', 'N/A')}"
        )

    print("\n" + "=" * 90)
    print("RESULTS COMPARISON")
    print("=" * 90)
    print(fmt(baseline_metrics, "Baseline (unmodified DenseNet)"))
    print(fmt(tuned_metrics,    "Head fine-tuned (this script)"))
    print("=" * 90)
    print(f"NF_conflict change: "
          f"{baseline_metrics.get('NF_conflict',{}).get('rate','?')} -> "
          f"{tuned_metrics.get('NF_conflict',{}).get('rate','?')}")
    print(f"Test images scored: {baseline_metrics.get('n_tested','?')}")
    print()

    # ── save full metrics JSON ─────────────────────────────────────────────────
    results = {
        "baseline": baseline_metrics,
        "head_finetuned": tuned_metrics,
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "train_n": len(train_df),
            "test_n":  len(test_df),
        },
    }
    results_path = args.output_dir / "lora_train_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full results saved: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
