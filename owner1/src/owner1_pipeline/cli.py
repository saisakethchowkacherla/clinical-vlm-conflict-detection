from __future__ import annotations

import argparse
import json
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def load_config(path: Path) -> PipelineConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    inputs = raw.get("inputs", {})
    outputs = raw.get("outputs", {})
    pipeline = raw.get("pipeline", {})

    missing = [
        key
        for key in ["chexpert_labels", "negbio_labels", "metadata", "split", "labevents"]
        if not inputs.get(key)
    ]
    if missing:
        raise ValueError(f"Missing required input path(s): {', '.join(missing)}")

    return PipelineConfig(
        chexpert_labels=Path(inputs["chexpert_labels"]),
        negbio_labels=Path(inputs["negbio_labels"]),
        metadata=Path(inputs["metadata"]),
        split=Path(inputs["split"]),
        labevents=Path(inputs["labevents"]),
        output_dir=Path(outputs.get("directory", "outputs")),
        finding=pipeline.get("finding", "Pneumonia"),
        lab_itemids=tuple(int(item) for item in pipeline.get("lab_itemids", [51301, 51300])),
        join_window_hours=float(pipeline.get("join_window_hours", 24)),
        kernel_per_cell=int(pipeline.get("kernel_per_cell", 120)),
        test_per_cell=int(pipeline.get("test_per_cell", 300)),
        random_seed=int(pipeline.get("random_seed", 20260604)),
        image_root=pipeline.get("image_root") or None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pneumonia/WBC manifests for the Owner 1 dataset pipeline."
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to TOML config.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
        summary = run_pipeline(config)
    except Exception as exc:  # pragma: no cover - keeps CLI failure readable
        print(f"owner1-build failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
