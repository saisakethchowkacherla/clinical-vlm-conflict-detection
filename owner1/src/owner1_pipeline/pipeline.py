from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


CONFIDENT_LABELS = {"pos", "neg"}
DEFAULT_WBC_ITEMIDS = (51301, 51300)


@dataclass(frozen=True)
class PipelineConfig:
    chexpert_labels: Path
    negbio_labels: Path
    metadata: Path
    split: Path
    labevents: Path
    output_dir: Path = Path("outputs")
    finding: str = "Pneumonia"
    lab_itemids: tuple[int, ...] = DEFAULT_WBC_ITEMIDS
    join_window_hours: float = 24.0
    kernel_per_cell: int = 120
    test_per_cell: int = 300
    random_seed: int = 20260604
    image_root: str | None = None


def normalize_label(value: object) -> str:
    if value is None:
        return "na"
    if isinstance(value, float) and math.isnan(value):
        return "na"
    text = str(value).strip()
    if text == "":
        return "na"
    try:
        numeric = float(text)
    except ValueError:
        return "na"
    if numeric == 1.0:
        return "pos"
    if numeric == 0.0:
        return "neg"
    if numeric == -1.0:
        return "uncertain"
    return "na"


def classify_lab_value(value: float, lower: float, upper: float) -> str:
    if pd.isna(value) or pd.isna(lower) or pd.isna(upper):
        raise ValueError("Lab value and reference range must be present.")
    if value > upper:
        return "high"
    if value < lower:
        return "low"
    return "normal"


def assign_cell(finding_label: str, lab_class: str) -> str:
    if finding_label == "pos" and lab_class in {"normal", "low"}:
        return "A"
    if finding_label == "neg" and lab_class == "high":
        return "B"
    if finding_label == "pos" and lab_class == "high":
        return "C"
    if finding_label == "neg" and lab_class in {"normal", "low"}:
        return "D"
    raise ValueError(f"No cell for finding={finding_label!r}, lab_class={lab_class!r}")


def parse_study_datetime(study_date: object, study_time: object) -> pd.Timestamp:
    date_text = str(study_date).strip()
    if date_text.endswith(".0"):
        date_text = date_text[:-2]
    date_text = date_text.zfill(8)

    time_text = "" if pd.isna(study_time) else str(study_time).strip()
    if time_text in {"", "nan", "NaN"}:
        time_text = "000000"
    time_text = time_text.split(".")[0].zfill(6)[:6]
    return pd.to_datetime(f"{date_text}{time_text}", format="%Y%m%d%H%M%S", errors="raise")


def image_relative_path(subject_id: int | str, study_id: int | str, dicom_id: str) -> str:
    subject = str(int(subject_id))
    study = str(int(study_id))
    return f"files/p{subject[:2]}/p{subject}/s{study}/{dicom_id}.jpg"


def load_agree_confident_labels(
    chexpert_path: Path, negbio_path: Path, finding: str
) -> pd.DataFrame:
    required = ["subject_id", "study_id", finding]
    chexpert = pd.read_csv(chexpert_path, usecols=required)
    negbio = pd.read_csv(negbio_path, usecols=required)

    chexpert = chexpert.rename(columns={finding: "chexpert_label"})
    negbio = negbio.rename(columns={finding: "negbio_label"})
    labels = chexpert.merge(negbio, on=["subject_id", "study_id"], how="inner")
    labels["chexpert_norm"] = labels["chexpert_label"].map(normalize_label)
    labels["negbio_norm"] = labels["negbio_label"].map(normalize_label)

    keep = (
        labels["chexpert_norm"].isin(CONFIDENT_LABELS)
        & labels["negbio_norm"].isin(CONFIDENT_LABELS)
        & (labels["chexpert_norm"] == labels["negbio_norm"])
    )
    labels = labels.loc[keep, ["subject_id", "study_id", "chexpert_norm"]].copy()
    return labels.rename(columns={"chexpert_norm": "finding_label"})


def load_split(split_path: Path) -> pd.DataFrame:
    split = pd.read_csv(split_path, usecols=["dicom_id", "study_id", "subject_id", "split"])
    return split.drop_duplicates()


def load_selected_metadata(metadata_path: Path, image_root: str | None = None) -> pd.DataFrame:
    base_cols = ["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"]
    # Peek at the CSV header to check if image_path is pre-built (e.g. COVID adapter)
    header = pd.read_csv(metadata_path, nrows=0).columns.tolist()
    usecols = base_cols + (["image_path"] if "image_path" in header else [])
    metadata = pd.read_csv(metadata_path, usecols=usecols)
    metadata["view_position"] = metadata["ViewPosition"].fillna("").astype(str).str.upper()
    metadata["view_rank"] = metadata["view_position"].map({"PA": 0, "AP": 1}).fillna(2).astype(int)
    metadata["study_datetime"] = [
        parse_study_datetime(date, time)
        for date, time in zip(metadata["StudyDate"], metadata["StudyTime"], strict=False)
    ]
    if "image_path" not in metadata.columns:
        metadata["image_path"] = [
            image_relative_path(subject_id, study_id, dicom_id)
            for subject_id, study_id, dicom_id in zip(
                metadata["subject_id"], metadata["study_id"], metadata["dicom_id"], strict=False
            )
        ]
        if image_root:
            root = Path(image_root)
            metadata["image_path"] = metadata["image_path"].map(lambda rel: str(root / rel))

    metadata = metadata.sort_values(
        ["subject_id", "study_id", "view_rank", "dicom_id"], kind="stable"
    )
    selected = metadata.drop_duplicates(["subject_id", "study_id"], keep="first").copy()
    return selected[
        [
            "subject_id",
            "study_id",
            "dicom_id",
            "view_position",
            "study_datetime",
            "image_path",
        ]
    ]


def load_wbc_labs(labevents_path: Path, itemids: Iterable[int]) -> pd.DataFrame:
    usecols = [
        "subject_id",
        "itemid",
        "charttime",
        "valuenum",
        "ref_range_lower",
        "ref_range_upper",
    ]
    itemids = set(int(item) for item in itemids)
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(labevents_path, usecols=usecols, chunksize=1_000_000):
        chunk = chunk.loc[chunk["itemid"].isin(itemids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["valuenum", "ref_range_lower", "ref_range_upper", "charttime"])
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime"])
        chunk["lab_class"] = [
            classify_lab_value(value, lower, upper)
            for value, lower, upper in zip(
                chunk["valuenum"],
                chunk["ref_range_lower"],
                chunk["ref_range_upper"],
                strict=False,
            )
        ]
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols + ["lab_class"])
    return pd.concat(chunks, ignore_index=True)


def join_nearest_lab(
    studies: pd.DataFrame, labs: pd.DataFrame, window_hours: float = 24.0
) -> pd.DataFrame:
    if studies.empty or labs.empty:
        return studies.iloc[0:0].copy()

    max_delta = pd.Timedelta(hours=window_hours)
    lab_groups = {
        subject_id: group.sort_values("charttime").reset_index(drop=True)
        for subject_id, group in labs.groupby("subject_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for study in studies.to_dict("records"):
        subject_labs = lab_groups.get(study["subject_id"])
        if subject_labs is None:
            continue

        deltas = (subject_labs["charttime"] - study["study_datetime"]).abs()
        nearest_index = deltas.idxmin()
        nearest_delta = deltas.loc[nearest_index]
        if nearest_delta > max_delta:
            continue

        lab = subject_labs.loc[nearest_index]
        row = dict(study)
        row.update(
            {
                "lab_itemid": int(lab["itemid"]),
                "lab_charttime": lab["charttime"],
                "lab_value": float(lab["valuenum"]),
                "lab_ref_lower": float(lab["ref_range_lower"]),
                "lab_ref_upper": float(lab["ref_range_upper"]),
                "lab_class": lab["lab_class"],
                "lab_delta_hours": nearest_delta.total_seconds() / 3600.0,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def build_manifest(config: PipelineConfig) -> pd.DataFrame:
    labels = load_agree_confident_labels(
        config.chexpert_labels, config.negbio_labels, config.finding
    )
    metadata = load_selected_metadata(config.metadata, config.image_root)
    split = load_split(config.split)
    labs = load_wbc_labs(config.labevents, config.lab_itemids)

    studies = labels.merge(metadata, on=["subject_id", "study_id"], how="inner")
    studies = studies.merge(
        split[["subject_id", "study_id", "dicom_id", "split"]],
        on=["subject_id", "study_id", "dicom_id"],
        how="left",
    )
    manifest = join_nearest_lab(studies, labs, config.join_window_hours)
    if manifest.empty:
        return manifest

    manifest["cell"] = [
        assign_cell(label, lab_class)
        for label, lab_class in zip(
            manifest["finding_label"], manifest["lab_class"], strict=False
        )
    ]
    manifest["finding"] = config.finding
    manifest = manifest.sort_values(["cell", "subject_id", "study_id"], kind="stable")
    return manifest.reset_index(drop=True)


def sample_patient_disjoint_balanced(
    manifest: pd.DataFrame,
    per_cell: int,
    seed: int,
    excluded_subjects: set[int] | None = None,
) -> pd.DataFrame:
    excluded_subjects = excluded_subjects or set()
    chosen_parts: list[pd.DataFrame] = []
    used_subjects: set[int] = set(excluded_subjects)

    for cell in ["A", "B", "C", "D"]:
        candidates = manifest.loc[
            (manifest["cell"] == cell) & (~manifest["subject_id"].isin(used_subjects))
        ].copy()
        candidates = candidates.sample(frac=1.0, random_state=seed + ord(cell))
        one_per_subject = candidates.drop_duplicates("subject_id", keep="first")
        if len(one_per_subject) < per_cell:
            raise ValueError(
                f"Not enough patient-disjoint rows for cell {cell}: "
                f"need {per_cell}, found {len(one_per_subject)}."
            )
        selected = one_per_subject.head(per_cell).copy()
        used_subjects.update(int(subject_id) for subject_id in selected["subject_id"])
        chosen_parts.append(selected)

    sample = pd.concat(chosen_parts, ignore_index=True)
    return sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def leakage_report(*samples: tuple[str, pd.DataFrame]) -> dict[str, object]:
    subject_sets = {
        name: set(int(subject_id) for subject_id in frame.get("subject_id", []))
        for name, frame in samples
    }
    overlaps: dict[str, int] = {}
    names = list(subject_sets)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(subject_sets[left] & subject_sets[right])
    return {
        "subjects_per_sample": {name: len(subjects) for name, subjects in subject_sets.items()},
        "pairwise_subject_overlaps": overlaps,
        "leakage_free": all(count == 0 for count in overlaps.values()),
    }


def cell_counts(manifest: pd.DataFrame, **samples: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {
        "manifest": manifest["cell"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0).to_dict()
        if not manifest.empty
        else {"A": 0, "B": 0, "C": 0, "D": 0}
    }
    for name, frame in samples.items():
        result[name] = (
            frame["cell"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0).to_dict()
            if not frame.empty
            else {"A": 0, "B": 0, "C": 0, "D": 0}
        )
    return result


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def write_json(data: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(config)
    manifest_path = config.output_dir / "kernel_manifest.csv"
    write_csv(manifest, manifest_path)

    samples: dict[str, pd.DataFrame] = {}
    sample_paths: dict[str, str] = {"kernel_manifest": str(manifest_path)}

    try:
        kernel = sample_patient_disjoint_balanced(
            manifest, config.kernel_per_cell, config.random_seed
        )
    except ValueError:
        kernel = pd.DataFrame()
    if not kernel.empty:
        kernel_path = config.output_dir / "kernel_480.csv"
        write_csv(kernel, kernel_path)
        samples["kernel_480"] = kernel
        sample_paths["kernel_480"] = str(kernel_path)

    excluded = set(int(subject_id) for subject_id in samples.get("kernel_480", pd.DataFrame()).get("subject_id", []))
    try:
        test = sample_patient_disjoint_balanced(
            manifest, config.test_per_cell, config.random_seed + 1000, excluded
        )
    except ValueError:
        test = pd.DataFrame()
    if not test.empty:
        test_path = config.output_dir / "kernel_test_1200.csv"
        write_csv(test, test_path)
        samples["kernel_test_1200"] = test
        sample_paths["kernel_test_1200"] = str(test_path)

    counts = cell_counts(manifest, **samples)
    counts_path = config.output_dir / "cell_counts.json"
    write_json(counts, counts_path)

    report = leakage_report(*samples.items()) if samples else leakage_report()
    report_path = config.output_dir / "leakage_report.json"
    write_json(report, report_path)

    return {
        "rows": int(len(manifest)),
        "cell_counts": counts,
        "leakage_report": report,
        "outputs": {**sample_paths, "cell_counts": str(counts_path), "leakage_report": str(report_path)},
    }
