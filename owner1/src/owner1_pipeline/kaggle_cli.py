from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .kaggle_adapter import KaggleAdapterConfig, run_kaggle_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build manifest from Kaggle COVID-19 Radiography Dataset."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Path to owner1/data/images/ (contains COVID-19_Radiography_Dataset/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_kaggle"),
        help="Directory to write manifests (default: outputs_kaggle).",
    )
    parser.add_argument("--kernel-per-cell", type=int, default=120)
    parser.add_argument("--test-per-cell", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument(
        "--no-covid",
        action="store_true",
        help="Exclude COVID images from pos class (use only Viral Pneumonia).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = KaggleAdapterConfig(
        images_dir=args.images_dir,
        kernel_per_cell=args.kernel_per_cell,
        test_per_cell=args.test_per_cell,
        random_seed=args.seed,
        include_covid=not args.no_covid,
    )
    try:
        summary = run_kaggle_pipeline(config, args.output_dir)
    except Exception as exc:
        print(f"kaggle-build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
