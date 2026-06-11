from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .metrics import score_predictions, write_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score Owner 2 clinical conflict predictions.")
    parser.add_argument("--manifest", required=True, type=Path, help="Owner 1 manifest CSV.")
    parser.add_argument("--predictions", required=True, type=Path, help="Model prediction CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Metrics JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = pd.read_csv(args.manifest)
        predictions = pd.read_csv(args.predictions)
        scored, metrics = score_predictions(manifest, predictions)
        write_outputs(scored, metrics, args.output)
    except Exception as exc:  # pragma: no cover
        print(f"owner2-score failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
