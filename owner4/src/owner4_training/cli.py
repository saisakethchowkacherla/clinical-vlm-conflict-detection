from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .mock_fix import run_mock_fix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Owner 4 mock training fix.")
    parser.add_argument("--predictions", required=True, type=Path, help="Input prediction CSV.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["conflict-aware-lora-mock", "naive-consistency-lora-mock"],
        help="Mock training strategy.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Fixed prediction CSV output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        fixed = run_mock_fix(args.predictions, args.strategy, args.output)
    except Exception as exc:  # pragma: no cover
        print(f"owner4-fix failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"rows": int(len(fixed)), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
