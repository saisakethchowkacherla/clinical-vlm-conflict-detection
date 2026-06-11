from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .gate import run_baseline_gate, gate_table
from .mock_model import make_mock_model

MOCK_MODELS = {
    "mock-medgemma-4b": {"flip_rate_conflict": 0.137, "flip_rate_control": 0.006, "seed": 20260604},
    "mock-medgemma-27b": {"flip_rate_conflict": 0.218, "flip_rate_control": 0.006, "seed": 20260605},
    "mock-fixed-lora": {"flip_rate_conflict": 0.05, "flip_rate_control": 0.006, "seed": 20260606},
    "mock-naive-lora": {"flip_rate_conflict": 0.02, "flip_rate_control": 0.12, "seed": 20260607},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Owner 3 inference harness — mock model runner.")
    parser.add_argument("--manifest", required=True, type=Path, help="Owner 1 manifest CSV.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to write predictions and gate table.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MOCK_MODELS.keys()),
        choices=list(MOCK_MODELS.keys()),
        help="Which mock models to run (default: all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        manifest = pd.read_csv(args.manifest)
    except Exception as exc:
        print(f"Failed to load manifest: {exc}", file=sys.stderr)
        return 1

    gate_rows: list[dict[str, object]] = []
    for name in args.models:
        cfg = MOCK_MODELS[name]
        model = make_mock_model(model_name=name, **cfg)  # type: ignore[arg-type]
        predictions = model.run_manifest_with_labels(manifest)

        pred_path = args.output_dir / f"predictions_{name}.csv"
        predictions.to_csv(pred_path, index=False)

        gate = run_baseline_gate(predictions, manifest)
        gate["model"] = name
        gate_rows.append(gate)
        print(f"  {name}: {'VALID' if gate['baseline_valid'] else 'DEGENERATE'} "
              f"(sens={gate['sensitivity']}, spec={gate['specificity']})")

    table = gate_table(gate_rows)
    table_path = args.output_dir / "gate_table.csv"
    table.to_csv(table_path, index=False)

    summary_path = args.output_dir / "gate_summary.json"
    summary_path.write_text(json.dumps(gate_rows, indent=2), encoding="utf-8")
    print(f"\nGate table: {table_path}")
    print(f"Predictions written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
