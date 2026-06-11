from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .data_builder import build_training_data, leakage_check, write_training_data
from .mock_trained_model import make_fixed_lora, make_naive_lora


# ── Build training data ──────────────────────────────────────────────────────

def build_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Owner 4: build LoRA training data from train pool manifest.")
    parser.add_argument("--train-manifest", required=True, type=Path, help="29,407-study training pool manifest CSV.")
    parser.add_argument("--test-manifest", required=True, type=Path, help="1,200-study held-out test manifest CSV (leakage check).")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write training CSVs and leakage report.")
    parser.add_argument("--defer-manifest", type=Path, default=None, help="Optional uncertain-study manifest for defer examples.")
    parser.add_argument("--no-naive", action="store_true", help="Omit naive-consistency examples.")
    return parser


def main_build(argv: list[str] | None = None) -> int:
    args = build_build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_manifest = pd.read_csv(args.train_manifest)
        test_manifest = pd.read_csv(args.test_manifest)
    except Exception as exc:
        print(f"Failed to load manifests: {exc}", file=sys.stderr)
        return 1

    defer_manifest = None
    if args.defer_manifest:
        try:
            defer_manifest = pd.read_csv(args.defer_manifest)
        except Exception as exc:
            print(f"Warning: could not load defer manifest: {exc}", file=sys.stderr)

    training_data = build_training_data(
        train_manifest,
        defer_manifest=defer_manifest,
        include_naive=not args.no_naive,
    )

    leak = leakage_check(training_data, test_manifest)
    if not leak["leakage_free"]:
        print(f"LEAKAGE DETECTED: {leak['overlap_count']} subjects overlap train/test!", file=sys.stderr)
        return 1

    faithful_path = args.output_dir / "training_faithful.csv"
    naive_path = args.output_dir / "training_naive_consistency.csv"
    all_path = args.output_dir / "training_all.csv"

    write_training_data(training_data[training_data["example_type"] == "faithful"], faithful_path)
    if not args.no_naive:
        write_training_data(training_data[training_data["example_type"] == "naive_consistency"], naive_path)
    write_training_data(training_data, all_path)

    leak_path = args.output_dir / "leakage_report.json"
    leak_path.write_text(json.dumps(leak, indent=2), encoding="utf-8")

    counts = training_data["example_type"].value_counts().to_dict()
    print(json.dumps({"rows": len(training_data), "by_type": counts, "leakage": leak}, indent=2))
    return 0


# ── Run mock trained model inference ────────────────────────────────────────

def build_infer_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Owner 4: run mock trained model on a manifest.")
    parser.add_argument("--manifest", required=True, type=Path, help="Test manifest CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write prediction CSVs.")
    parser.add_argument(
        "--model",
        choices=["fixed-lora", "naive-lora", "both"],
        default="both",
        help="Which mock model to run.",
    )
    return parser


def main_infer(argv: list[str] | None = None) -> int:
    args = build_infer_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = pd.read_csv(args.manifest)
    except Exception as exc:
        print(f"Failed to load manifest: {exc}", file=sys.stderr)
        return 1

    models = []
    if args.model in {"fixed-lora", "both"}:
        models.append(("fixed_lora", make_fixed_lora()))
    if args.model in {"naive-lora", "both"}:
        models.append(("naive_consistency_lora", make_naive_lora()))

    for name, model in models:
        predictions = model.run_manifest_with_labels(manifest)
        out_path = args.output_dir / f"predictions_{name}.csv"
        predictions.to_csv(out_path, index=False)
        print(f"  {model.model_name}: predictions -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_build())
