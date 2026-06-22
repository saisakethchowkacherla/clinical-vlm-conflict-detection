"""
Dedicated 4-way COVID image classifier.

Difference from lora_train.py
------------------------------
lora_train.py fine-tunes the existing 18-pathology head toward a binary
"finding present/absent" proxy (the composite score). It does not learn
to distinguish COVID from Viral Pneumonia or Lung Opacity directly.

This script instead trains a brand-new classification head directly on
the real ground-truth class labels (COVID / Normal / Viral Pneumonia /
Lung_Opacity) recovered from the manifest's image_path. Same frozen-
feature-extraction strategy as lora_train.py: all convolutional layers
stay frozen, only a new nn.Linear(1024, 4) head is trained.

Usage
-----
  python -m owner4_training.covid_classifier_train \\
      --manifest   owner1/outputs_covid/kernel_manifest.csv \\
      --output-dir owner4/outputs_covid \\
      --epochs     8 \\
      --max-train  600 \\
      --max-test   300
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_train import load_base_model, preprocess_image

warnings.filterwarnings("ignore", category=UserWarning)

CLASSES = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def class_from_path(image_path: str) -> str:
    """Recover the true 4-way class from the COVID dataset folder structure:
    .../COVID-19_Radiography_Dataset/<Class>/images/<file>.png
    """
    parts = Path(image_path).parts
    idx = parts.index("images")
    return parts[idx - 1]


def extract_features_batch(
    model, image_paths: list[str], verbose: bool = True
) -> tuple[torch.Tensor, list[int]]:
    """
    Returns (features, valid_indices). valid_indices lists which positions in
    image_paths succeeded — missing/corrupt files are skipped, not fatal.
    """
    features_list: list[torch.Tensor] = []
    valid_indices: list[int] = []
    skipped: list[str] = []
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        if not Path(path).exists():
            skipped.append(path)
            continue
        try:
            tensor = preprocess_image(path)
        except Exception as exc:  # corrupt/unreadable image
            skipped.append(f"{path} ({exc})")
            continue
        with torch.no_grad():
            feat = model.features(tensor)
            feat = F.relu(feat, inplace=True)
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            feat = feat.view(feat.size(0), -1)
        features_list.append(feat)
        valid_indices.append(i)
        if verbose and (i + 1) % 100 == 0:
            print(f"  feature extraction: {i + 1}/{n} images", flush=True)

    if skipped:
        print(f"  WARNING: skipped {len(skipped)} unreadable/missing image(s):")
        for s in skipped[:10]:
            print(f"    {s}")
        if len(skipped) > 10:
            print(f"    ... and {len(skipped) - 10} more")

    return torch.cat(features_list, dim=0), valid_indices


def train_covid_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Linear:
    head = nn.Linear(1024, len(CLASSES))
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = len(labels)
    perm = torch.randperm(n)
    features = features[perm]
    labels = labels[perm]

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        batches = 0
        for start in range(0, n, batch_size):
            feats = features[start : start + batch_size]
            lbls  = labels[start : start + batch_size]

            logits = head(feats)
            loss = criterion(logits, lbls)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (logits.argmax(dim=1) == lbls).sum().item()
            batches += 1

        acc = correct / n
        print(f"  epoch {epoch}/{epochs}  avg_loss={epoch_loss/batches:.4f}  train_acc={acc:.4f}", flush=True)

    return head


def evaluate(head: nn.Linear, features: torch.Tensor, labels: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = head(features)
        preds = logits.argmax(dim=1)

    correct = (preds == labels).sum().item()
    n = len(labels)
    accuracy = correct / n

    confusion = {true_c: {pred_c: 0 for pred_c in CLASSES} for true_c in CLASSES}
    for true_idx, pred_idx in zip(labels.tolist(), preds.tolist()):
        confusion[CLASSES[true_idx]][CLASSES[pred_idx]] += 1

    per_class = {}
    for c in CLASSES:
        true_mask = labels == CLASS_TO_IDX[c]
        n_c = true_mask.sum().item()
        correct_c = ((preds == labels) & true_mask).sum().item()
        per_class[c] = {
            "n": n_c,
            "correct": correct_c,
            "accuracy": round(correct_c / n_c, 4) if n_c else None,
        }

    return {
        "n": n,
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a dedicated 4-way COVID image classifier.")
    parser.add_argument("--manifest",   required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("owner4/outputs_covid"), type=Path)
    parser.add_argument("--epochs",     default=8,   type=int)
    parser.add_argument("--max-train",  default=None, type=int)
    parser.add_argument("--max-test",   default=None, type=int)
    parser.add_argument("--lr",         default=1e-3, type=float)
    parser.add_argument("--batch-size", default=64,   type=int)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    manifest["true_class"] = manifest["image_path"].map(class_from_path)
    print(f"Manifest loaded: {len(manifest):,} rows")
    print(f"True class distribution:\n{manifest['true_class'].value_counts().to_string()}")

    train_df = manifest.loc[manifest["split"] == "train"].copy().reset_index(drop=True)
    test_df  = manifest.loc[manifest["split"] == "test"].copy().reset_index(drop=True)

    if args.max_train and len(train_df) > args.max_train:
        train_df = train_df.sample(args.max_train, random_state=20260604).reset_index(drop=True)
    if args.max_test and len(test_df) > args.max_test:
        test_df = test_df.sample(args.max_test, random_state=20260604).reset_index(drop=True)

    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    print(f"Train class distribution:\n{train_df['true_class'].value_counts().to_string()}")
    print(f"Test class distribution:\n{test_df['true_class'].value_counts().to_string()}")

    print("\nLoading DenseNet121 (feature extractor only, frozen)...")
    model = load_base_model()

    print(f"\n[FEATURE EXTRACTION] {len(train_df):,} training images...")
    t0 = time.time()
    train_features, train_valid = extract_features_batch(model, train_df["image_path"].tolist())
    train_df = train_df.iloc[train_valid].reset_index(drop=True)
    train_labels = torch.tensor([CLASS_TO_IDX[c] for c in train_df["true_class"]], dtype=torch.long)
    print(f"  done in {time.time()-t0:.1f}s  ({len(train_df):,} usable images)")

    print(f"\n[FEATURE EXTRACTION] {len(test_df):,} test images...")
    t0 = time.time()
    test_features, test_valid = extract_features_batch(model, test_df["image_path"].tolist())
    test_df = test_df.iloc[test_valid].reset_index(drop=True)
    test_labels = torch.tensor([CLASS_TO_IDX[c] for c in test_df["true_class"]], dtype=torch.long)
    print(f"  done in {time.time()-t0:.1f}s  ({len(test_df):,} usable images)")

    print(f"\n[TRAINING] {args.epochs} epochs, lr={args.lr}, classes={CLASSES}")
    t0 = time.time()
    head = train_covid_head(train_features, train_labels, args.epochs, args.batch_size, args.lr)
    print(f"  training done in {time.time()-t0:.1f}s")

    weights_path = args.output_dir / "covid_4way_head.pt"
    torch.save(head.state_dict(), weights_path)
    print(f"  saved: {weights_path}")

    print("\n[EVALUATION] Scoring on held-out test set...")
    results = evaluate(head, test_features, test_labels)

    print("\n" + "=" * 70)
    print(f"Overall test accuracy: {results['accuracy']:.4f}  (n={results['n']})")
    print("=" * 70)
    print("\nPer-class accuracy:")
    for c, stats in results["per_class"].items():
        print(f"  {c:<18} n={stats['n']:<5} acc={stats['accuracy']}")

    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "  " + " " * 18 + "".join(f"{c[:8]:>10}" for c in CLASSES)
    print(header)
    for true_c in CLASSES:
        row = "  " + f"{true_c:<18}" + "".join(f"{results['confusion_matrix'][true_c][p]:>10}" for p in CLASSES)
        print(row)

    results_path = args.output_dir / "covid_4way_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results saved: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
