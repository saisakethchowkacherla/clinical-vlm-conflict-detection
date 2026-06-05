from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gates import run_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply Owner 5 inference gate.")
    parser.add_argument("--predictions", required=True, type=Path, help="Input prediction CSV.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["defer-on-disagree"],
        help="Inference gate strategy.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Gated prediction CSV output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        gated = run_gate(args.predictions, args.strategy, args.output)
    except Exception as exc:  # pragma: no cover
        print(f"owner5-gate failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"rows": int(len(gated)), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
