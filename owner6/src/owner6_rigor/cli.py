from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aggregate import write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Owner 2 metrics for Owner 6.")
    parser.add_argument("--metrics", required=True, nargs="+", type=Path, help="Metrics JSON file(s).")
    parser.add_argument("--output", required=True, type=Path, help="Markdown summary output.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = write_summary(args.metrics, args.output)
    except Exception as exc:  # pragma: no cover
        print(f"owner6-aggregate failed: {exc}", file=sys.stderr)
        return 1
    print(summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
