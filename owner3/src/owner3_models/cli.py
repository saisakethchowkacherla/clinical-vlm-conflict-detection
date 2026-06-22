from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .mock_inference import PROFILES, run_mock_inference

REAL_MODEL = "densenet121-xrv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Owner 3 model inference.")
    parser.add_argument("--manifest", required=True, type=Path, help="Owner 1 manifest CSV.")
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(PROFILES) + [REAL_MODEL],
        help="Model to use. Use 'densenet121-xrv' for real torchxrayvision inference.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Prediction CSV output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.model == REAL_MODEL:
            from .real_inference import run_real_inference
            predictions = run_real_inference(args.manifest, args.output)
        else:
            predictions = run_mock_inference(args.manifest, args.model, args.output)
    except Exception as exc:
        print(f"owner3-infer failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"rows": int(len(predictions)), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
