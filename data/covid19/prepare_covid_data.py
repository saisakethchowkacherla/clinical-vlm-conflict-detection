"""
Convert the COVID-19 Radiography Dataset into the five MIMIC-shaped CSVs
the Owner 1 pipeline expects.

Classes used:
  COVID           -> finding = 1.0  (positive)
  Normal          -> finding = 0.0  (negative)
  Viral Pneumonia -> finding = 1.0  (positive)
  Lung_Opacity    -> finding = 1.0  (positive — radiological abnormality)

WBC assignment strategy (hybrid):
  COVID / Viral Pneumonia / Lung Opacity — sampled from per-class clinical
    distributions (see sample_wbc_for_class). Cell assignment follows from
    the sampled value, not forced ahead of time.
  Normal — alternating fixed values (7.0 / 15.0) so ~50% of Normal images
    get high WBC, keeping Cell B populated. Realistic sampling would clip
    Normal to <= ref_high = 11.0, leaving Cell B empty and breaking the
    balanced benchmark.

Output files written to the same directory as this script:
  chexpert.csv   negbio.csv   metadata.csv   split.csv   labevents.csv

Run from any location:
  python data/covid19/prepare_covid_data.py
"""
from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

SEED = 20260604
WBC_ITEMID = 51301
WBC_REF_LOW = 4.5
WBC_REF_HIGH = 11.0

DATASET_ROOT = Path(__file__).parent.parent / "covid19-radiography-database" / "COVID-19_Radiography_Dataset"
OUT_DIR = Path(__file__).parent
BASE_DATE = datetime(2020, 1, 1)


def sample_wbc_for_class(class_name: str, rng: np.random.Generator) -> float:
    """
    Sample a WBC value from a per-class distribution that reflects
    real-world clinical patterns.

      Normal         — genuinely healthy WBC range
      COVID          — leukopenia tendency observed in COVID-19
      Viral Pneumonia— wide spread, normal or mildly elevated
      Lung_Opacity   — bimodal: 50% normal causes, 50% infectious/elevated
    """
    if class_name == "Normal":
        # Not reached — Normal WBC is assigned in build_records using the
        # alternating rule to preserve Cell B cases (see comment there).
        raise NotImplementedError("Normal WBC is handled separately in build_records.")

    if class_name == "COVID":
        v = rng.normal(5.0, 1.2)
        return float(np.clip(v, 3.0, 8.0))

    if class_name == "Viral Pneumonia":
        v = rng.normal(8.0, 2.5)
        return float(np.clip(v, 4.0, 15.0))

    if class_name in ("Lung_Opacity", "Lung Opacity"):
        if rng.random() < 0.5:
            # Non-infectious opacity — normal WBC
            v = rng.normal(7.5, 1.5)
            return float(np.clip(v, 4.5, 11.0))
        else:
            # Infectious / inflammatory opacity — elevated WBC
            v = rng.normal(14.0, 2.5)
            return float(np.clip(v, 11.0, 20.0))

    raise ValueError(f"Unknown class: {class_name!r}")


def assign_cell(finding: float, wbc_value: float) -> str:
    """Derive cell from (image finding, sampled WBC) pair."""
    lab_high = wbc_value > WBC_REF_HIGH
    if finding == 1.0:
        return "C" if lab_high else "A"
    else:
        return "B" if lab_high else "D"


def collect_images(class_name: str, finding_value: float) -> list[dict]:
    img_dir = DATASET_ROOT / class_name / "images"
    rows = []
    for img in sorted(img_dir.glob("*.png")):
        rows.append({
            "class": class_name,
            "filename": img.stem,
            "image_path": str(img),
            "finding": finding_value,
        })
    return rows


def build_records(rng_py: random.Random, rng_np: np.random.Generator) -> list[dict]:
    positive = (
        collect_images("COVID", 1.0)
        + collect_images("Viral Pneumonia", 1.0)
        + collect_images("Lung_Opacity", 1.0)
    )
    negative = collect_images("Normal", 0.0)

    rng_py.shuffle(positive)
    rng_py.shuffle(negative)

    records = []
    for subject_id, entry in enumerate(positive + negative, start=1):
        entry["subject_id"] = subject_id
        entry["study_id"] = subject_id
        entry["dicom_id"] = f"d{subject_id:07d}"
        entry["study_datetime"] = BASE_DATE + timedelta(days=(subject_id - 1) % 365)
        entry["lab_charttime"] = entry["study_datetime"] + timedelta(hours=2)

        if entry["class"] == "Normal":
            # Alternating fixed values so ~50% of Normal images get high WBC,
            # populating Cell B (neg finding + high WBC). Realistic sampling
            # clips Normal to ≤ ref_high = 11.0, leaving Cell B empty.
            wbc_value = 15.0 if subject_id % 2 == 0 else 7.0
        else:
            wbc_value = sample_wbc_for_class(entry["class"], rng_np)
        entry["wbc_value"] = round(wbc_value, 3)
        entry["cell"] = assign_cell(entry["finding"], wbc_value)
        records.append(entry)
    return records


def print_distribution_stats(records: list[dict]) -> None:
    """Print WBC sampling stats per class so the values can be verified."""
    classes = sorted({r["class"] for r in records})
    print("\nWBC distribution stats per class:")
    print(f"  {'Class':<20} {'N':>6} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}  Cell counts")
    for cls in classes:
        wbcs = [r["wbc_value"] for r in records if r["class"] == cls]
        arr = np.array(wbcs)
        cells = {c: sum(1 for r in records if r["class"] == cls and r["cell"] == c)
                 for c in "ABCD"}
        cell_str = "  ".join(f"{c}:{cells[c]}" for c in "ABCD" if cells[c])
        print(
            f"  {cls:<20} {len(wbcs):>6} {arr.mean():>7.2f} {arr.std():>7.2f}"
            f" {arr.min():>7.2f} {arr.max():>7.2f}  {cell_str}"
        )

    print("\nOverall cell counts (all classes combined):")
    total_cells = {c: sum(1 for r in records if r["cell"] == c) for c in "ABCD"}
    for c in "ABCD":
        print(f"  Cell {c}: {total_cells[c]:,}")

    empty = [c for c in "ABCD" if total_cells[c] == 0]
    if empty:
        print(f"\n  WARNING: Cell(s) {', '.join(empty)} have 0 cases.")
        print("  The balanced kernel_480 / kernel_test_1200 sampling will fail.")
        print("  This is expected — the per-class WBC distributions do not")
        print("  produce negative-finding + high-WBC combinations (cell B)")
        print("  because Normal images are clipped to WBC <= ref_high = 11.0.")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>6,} rows -> {path}")


def main() -> None:
    rng_py = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    print("Scanning COVID-19 Radiography Dataset...")
    records = build_records(rng_py, rng_np)
    print(f"  {len(records):,} total images (COVID + Viral Pneumonia + Lung Opacity + Normal)")

    print_distribution_stats(records)
    print()

    # ── chexpert.csv & negbio.csv ─────────────────────────────────────────────
    label_rows = [
        {"subject_id": r["subject_id"], "study_id": r["study_id"], "COVID": r["finding"]}
        for r in records
    ]
    write_csv(OUT_DIR / "chexpert.csv", ["subject_id", "study_id", "COVID"], label_rows)
    write_csv(OUT_DIR / "negbio.csv",   ["subject_id", "study_id", "COVID"], label_rows)

    # ── metadata.csv ──────────────────────────────────────────────────────────
    meta_rows = [
        {
            "dicom_id":    r["dicom_id"],
            "subject_id":  r["subject_id"],
            "study_id":    r["study_id"],
            "ViewPosition": "PA",
            "StudyDate":   r["study_datetime"].strftime("%Y%m%d"),
            "StudyTime":   r["study_datetime"].strftime("%H%M%S"),
            "image_path":  r["image_path"],
        }
        for r in records
    ]
    write_csv(
        OUT_DIR / "metadata.csv",
        ["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime", "image_path"],
        meta_rows,
    )

    # ── split.csv ─────────────────────────────────────────────────────────────
    split_rows = [
        {
            "dicom_id":   r["dicom_id"],
            "subject_id": r["subject_id"],
            "study_id":   r["study_id"],
            "split":      "train" if r["subject_id"] % 5 != 0 else "test",
        }
        for r in records
    ]
    write_csv(OUT_DIR / "split.csv", ["dicom_id", "subject_id", "study_id", "split"], split_rows)

    # ── labevents.csv ─────────────────────────────────────────────────────────
    lab_rows = [
        {
            "subject_id":       r["subject_id"],
            "itemid":           WBC_ITEMID,
            "charttime":        r["lab_charttime"].strftime("%Y-%m-%d %H:%M:%S"),
            "valuenum":         r["wbc_value"],
            "ref_range_lower":  WBC_REF_LOW,
            "ref_range_upper":  WBC_REF_HIGH,
        }
        for r in records
    ]
    write_csv(
        OUT_DIR / "labevents.csv",
        ["subject_id", "itemid", "charttime", "valuenum", "ref_range_lower", "ref_range_upper"],
        lab_rows,
    )

    pos = sum(1 for r in records if r["finding"] == 1.0)
    neg = len(records) - pos
    print(f"\nDone. {pos:,} positive (COVID + Viral Pneumonia + Lung Opacity), {neg:,} negative (Normal).")


if __name__ == "__main__":
    main()
