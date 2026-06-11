#!/usr/bin/env python3
"""
Generate a full-scale synthetic dataset that mirrors MIMIC-CXR + MIMIC-IV structure.

Output files (plain CSV, no compression — pandas reads both):
  chexpert.csv    -- CheXpert finding labels (subject_id, study_id, Pneumonia)
  negbio.csv      -- NegBio finding labels   (subject_id, study_id, Pneumonia)
  metadata.csv    -- Study metadata          (dicom_id, subject_id, study_id,
                                              ViewPosition, StudyDate, StudyTime)
  split.csv       -- Train/test split        (dicom_id, subject_id, study_id, split)
  labevents.csv   -- WBC lab events          (subject_id, itemid, charttime,
                                              valuenum, ref_range_lower, ref_range_upper)

Scale: 500 studies per cell (A/B/C/D) = 2,000 valid joined studies.
Supports kernel_per_cell=120 (480) + test_per_cell=300 (1,200) with ~80/cell buffer.

Noise studies (filtered by the pipeline):
  - 50 with uncertain / disagreeing labels  -> dropped by agree-confident filter
  - 50 with no WBC within ±24h             -> dropped by join_nearest_lab
  - 50 with wrong lab itemid               -> dropped by load_wbc_labs itemid filter
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

# ── constants ──────────────────────────────────────────────────────────────────
SEED = 20260604
N_PER_CELL = 500
CELLS = ["A", "B", "C", "D"]
FINDING = "Pneumonia"
WBC_ITEMIDS = [51301, 51300]
REF_LOWER = 4.0
REF_UPPER = 11.0
STUDY_DATE = 22000101       # all studies on the same synthetic date (YYYYMMDD)
OUTPUT_DIR = Path(__file__).parent

# Cell definitions (from EXPERIMENT_SPEC.md §2)
#   A  pos  + normal/low WBC  -> conflict
#   B  neg  + high WBC        -> conflict
#   C  pos  + high WBC        -> corroborate
#   D  neg  + normal/low WBC  -> corroborate
FINDING_FOR_CELL = {"A": 1, "B": 0, "C": 1, "D": 0}


# ── helpers ────────────────────────────────────────────────────────────────────

def gen_wbc(cell: str, rng: random.Random) -> float:
    """Return a WBC value (K/uL, 2 dp) consistent with the cell definition."""
    if cell in ("A", "D"):                      # normal or low
        if rng.random() < 0.70:
            return round(rng.uniform(4.00, 10.90), 2)   # normal  [4, 11]
        else:
            return round(rng.uniform(1.50, 3.99), 2)    # low     [<4]
    else:                                       # B, C — high
        return round(rng.uniform(11.10, 28.00), 2)      # high    [>11]


def gen_study_time(rng: random.Random) -> int:
    """Random HHMMSS integer in business hours (06–20)."""
    h = rng.randint(6, 20)
    m = rng.randint(0, 59)
    s = rng.randint(0, 59)
    return h * 10000 + m * 100 + s


def hhmmss_to_minutes(t: int) -> int:
    return (t // 10000) * 60 + (t % 10000) // 100


def minutes_to_hms(mins: int) -> tuple[int, int, int]:
    mins = max(0, min(23 * 60 + 59, mins))
    return mins // 60, (mins % 60), 0


def gen_lab_charttime(study_time_int: int, rng: random.Random, offset_h: int = 6) -> str:
    """Lab charttime within ±offset_h hours of study time, same synthetic date."""
    study_min = hhmmss_to_minutes(study_time_int)
    offset_min = rng.randint(-offset_h * 60, offset_h * 60)
    h, m, s = minutes_to_hms(study_min + offset_min)
    return f"2200-01-01 {h:02d}:{m:02d}:{s:02d}"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {len(rows):>6} rows  ->  {path.name}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = random.Random(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. valid studies (500 per cell) ───────────────────────────────────────
    valid: list[tuple[int, int, str]] = []          # (subject_id, study_id, cell)
    for cell_idx, cell in enumerate(CELLS):
        for i in range(N_PER_CELL):
            subject_id = 10000001 + cell_idx * N_PER_CELL + i
            study_id   = 50000001 + cell_idx * N_PER_CELL + i
            valid.append((subject_id, study_id, cell))

    # ── 2. noise studies ──────────────────────────────────────────────────────
    noise_offset = len(valid)
    noise: list[tuple[int, int, str]] = []
    for i in range(150):
        subject_id = 10000001 + noise_offset + i
        study_id   = 50000001 + noise_offset + i
        kind = "uncertain" if i < 50 else ("no_lab" if i < 100 else "wrong_itemid")
        noise.append((subject_id, study_id, kind))

    # ── 3. chexpert / negbio ──────────────────────────────────────────────────
    chexpert_rows: list[dict] = []
    negbio_rows:   list[dict] = []

    for subject_id, study_id, cell in valid:
        lbl = FINDING_FOR_CELL[cell]
        chexpert_rows.append({"subject_id": subject_id, "study_id": study_id, FINDING: lbl})
        negbio_rows  .append({"subject_id": subject_id, "study_id": study_id, FINDING: lbl})

    for subject_id, study_id, kind in noise:
        if kind == "uncertain":
            # Ensure at least one labeler is uncertain (-1) or they disagree.
            cx = rng.choice([-1, 1, 0])
            nb = rng.choice([-1, 1, 0])
            if cx == nb and cx != -1:
                cx = -1                             # force uncertainty
        else:
            # Valid, confident, agreeing labels — excluded for a different reason.
            lbl = rng.choice([0, 1])
            cx = nb = lbl
        chexpert_rows.append({"subject_id": subject_id, "study_id": study_id, FINDING: cx})
        negbio_rows  .append({"subject_id": subject_id, "study_id": study_id, FINDING: nb})

    # ── 4. metadata ───────────────────────────────────────────────────────────
    metadata_rows: list[dict] = []
    study_time_by_subject: dict[int, int] = {}

    for subject_id, study_id, _ in valid + noise:
        study_time = gen_study_time(rng)
        study_time_by_subject[subject_id] = study_time

        # Primary view — PA preferred (matches EXPERIMENT_SPEC.md §2 frontal-view rule)
        primary_view = rng.choice(["PA", "PA", "AP"])
        dicom_id = f"d{study_id}"
        metadata_rows.append({
            "dicom_id":      dicom_id,
            "subject_id":    subject_id,
            "study_id":      study_id,
            "ViewPosition":  primary_view,
            "StudyDate":     STUDY_DATE,
            "StudyTime":     study_time,
        })
        # ~30 % of studies also have a second view (like real MIMIC)
        if rng.random() < 0.30:
            alt_view = "AP" if primary_view == "PA" else "PA"
            metadata_rows.append({
                "dicom_id":      f"d{study_id}_alt",
                "subject_id":    subject_id,
                "study_id":      study_id,
                "ViewPosition":  alt_view,
                "StudyDate":     STUDY_DATE,
                "StudyTime":     study_time,
            })

    # ── 5. split ──────────────────────────────────────────────────────────────
    # The pipeline left-joins split as metadata only; "train"/"test" values are
    # carried through but never used as a filter here.
    split_rows: list[dict] = []
    for row in metadata_rows:
        split_rows.append({
            "dicom_id":   row["dicom_id"],
            "subject_id": row["subject_id"],
            "study_id":   row["study_id"],
            "split":      "train" if rng.random() < 0.80 else "test",
        })

    # ── 6. labevents ──────────────────────────────────────────────────────────
    labevents_rows: list[dict] = []

    # Valid studies: one WBC within ±6h of the study.
    for subject_id, study_id, cell in valid:
        labevents_rows.append({
            "subject_id":      subject_id,
            "itemid":          rng.choice(WBC_ITEMIDS),
            "charttime":       gen_lab_charttime(study_time_by_subject[subject_id], rng, offset_h=6),
            "valuenum":        gen_wbc(cell, rng),
            "ref_range_lower": REF_LOWER,
            "ref_range_upper": REF_UPPER,
        })

    for subject_id, study_id, kind in noise:
        if kind == "uncertain":
            # Valid WBC — study drops because of label disagreement, not missing lab.
            labevents_rows.append({
                "subject_id":      subject_id,
                "itemid":          rng.choice(WBC_ITEMIDS),
                "charttime":       gen_lab_charttime(study_time_by_subject[subject_id], rng, offset_h=6),
                "valuenum":        round(rng.uniform(4.00, 15.00), 2),
                "ref_range_lower": REF_LOWER,
                "ref_range_upper": REF_UPPER,
            })
        elif kind == "no_lab":
            # Lab exists but is 48h later — outside the ±24h window.
            labevents_rows.append({
                "subject_id":      subject_id,
                "itemid":          rng.choice(WBC_ITEMIDS),
                "charttime":       "2200-01-03 12:00:00",
                "valuenum":        round(rng.uniform(4.00, 15.00), 2),
                "ref_range_lower": REF_LOWER,
                "ref_range_upper": REF_UPPER,
            })
        else:   # wrong_itemid
            # Lab is timely but itemid is not 51300/51301 — filtered by load_wbc_labs.
            labevents_rows.append({
                "subject_id":      subject_id,
                "itemid":          99999,
                "charttime":       gen_lab_charttime(study_time_by_subject[subject_id], rng, offset_h=6),
                "valuenum":        round(rng.uniform(4.00, 15.00), 2),
                "ref_range_lower": REF_LOWER,
                "ref_range_upper": REF_UPPER,
            })

    # Shuffle so itemid clusters aren't sequential (mirrors real labevents).
    rng.shuffle(labevents_rows)

    # ── 7. write files ────────────────────────────────────────────────────────
    print(f"\nWriting synthetic data to {OUTPUT_DIR}/\n")
    write_csv(OUTPUT_DIR / "chexpert.csv",  chexpert_rows,  ["subject_id", "study_id", FINDING])
    write_csv(OUTPUT_DIR / "negbio.csv",    negbio_rows,    ["subject_id", "study_id", FINDING])
    write_csv(OUTPUT_DIR / "metadata.csv",  metadata_rows,
              ["dicom_id", "subject_id", "study_id", "ViewPosition", "StudyDate", "StudyTime"])
    write_csv(OUTPUT_DIR / "split.csv",     split_rows,
              ["dicom_id", "subject_id", "study_id", "split"])
    write_csv(OUTPUT_DIR / "labevents.csv", labevents_rows,
              ["subject_id", "itemid", "charttime", "valuenum", "ref_range_lower", "ref_range_upper"])

    n_valid = len(valid)
    n_noise = len(noise)
    print(f"\nDone.")
    print(f"  {n_valid} valid studies  ({N_PER_CELL}/cell × {len(CELLS)} cells)")
    print(f"  {n_noise} noise studies  (50 uncertain-label, 50 no-lab, 50 wrong-itemid)")
    print(f"\nExpected pipeline manifest rows : {n_valid}")
    print(f"Expected kernel sample (480)    : 120/cell × 4 cells")
    print(f"Expected test sample  (1,200)   : 300/cell × 4 cells")


if __name__ == "__main__":
    main()
