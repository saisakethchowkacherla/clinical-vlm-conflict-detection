from __future__ import annotations

import json
from pathlib import Path


def load_metric(path: Path) -> dict[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source"] = str(path)
    return data


def pct(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100:.1f}%"


def summarize_metrics(paths: list[Path]) -> str:
    rows = []
    for path in paths:
        metric = load_metric(path)
        baseline = metric["baseline"]
        nf = metric["negative_flip"]
        rows.append(
            {
                "source": path.name,
                "rows": metric["rows_scored"],
                "baseline_valid": baseline["baseline_valid"],
                "accuracy": baseline["accuracy"]["rate"],
                "nf_conflict": nf["conflict_A_B"]["rate"],
                "nf_control": nf["control_C_D"]["rate"],
                "gap": nf["gap"],
            }
        )

    lines = [
        "# Aggregated Metrics",
        "",
        "| Source | Rows | Baseline Valid | Accuracy | NF Conflict | NF Control | Gap |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {source} | {rows} | {baseline_valid} | {accuracy} | {nf_conflict} | {nf_control} | {gap} |".format(
                source=row["source"],
                rows=row["rows"],
                baseline_valid=row["baseline_valid"],
                accuracy=pct(row["accuracy"]),
                nf_conflict=pct(row["nf_conflict"]),
                nf_control=pct(row["nf_control"]),
                gap=pct(row["gap"]),
            )
        )
    lines.append("")
    lines.append("Use this as a writing table, not as final paper statistics.")
    return "\n".join(lines) + "\n"


def write_summary(paths: list[Path], output_path: Path) -> str:
    summary = summarize_metrics(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    return summary
