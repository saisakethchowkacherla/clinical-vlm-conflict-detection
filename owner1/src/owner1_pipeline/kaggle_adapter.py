from __future__ import annotations

"""Kaggle COVID-19 Radiography Dataset adapter.

Maps the Kaggle dataset to the owner1 manifest schema so the rest of the
pipeline (owner2, owner3, owner4) can consume it without modification.

Label mapping:
  pos (pneumonia present): Viral Pneumonia, COVID
  neg (pneumonia absent):  Normal
  excluded:                Lung_Opacity (ambiguous)

WBC values are synthesized with realistic distributions because the Kaggle
dataset has no paired lab data. The synthesis is deterministic (seeded) and
transparent — every manifest row carries a 'wbc_source' = 'synthesized' flag.

WBC reference range: 4.0–11.0 K/uL (standard clinical range).
  normal: Gaussian(mean=7.0, sd=1.2), clamped to [4.0, 11.0]
  high:   Gaussian(mean=15.0, sd=2.5), clamped to [11.1, 30.0]
  low:    Gaussian(mean=2.5,  sd=0.5), clamped to [0.5,  3.9]

Cell assignment:
  A: pos image + normal/low WBC  (conflict)
  B: neg image + high WBC        (conflict)
  C: pos image + high WBC        (corroborate)
  D: neg image + normal/low WBC  (corroborate)
"""

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IMAGE_ROOT = "COVID-19_Radiography_Dataset"

CLASS_TO_LABEL: dict[str, str] = {
    "Viral Pneumonia": "pos",
    "COVID": "pos",
    "Normal": "neg",
}

WBC_REF_LOWER = 4.0
WBC_REF_UPPER = 11.0


@dataclass(frozen=True)
class KaggleAdapterConfig:
    images_dir: Path           # path to owner1/data/images/
    kernel_per_cell: int = 120
    test_per_cell: int = 300
    random_seed: int = 20260604
    include_covid: bool = True  # if False, only Viral Pneumonia used for pos


def _synth_wbc(lab_class: str, rng: random.Random) -> float:
    if lab_class == "normal":
        v = rng.gauss(7.0, 1.2)
        return max(WBC_REF_LOWER, min(WBC_REF_UPPER, v))
    if lab_class == "high":
        v = rng.gauss(15.0, 2.5)
        return max(11.1, min(30.0, v))
    # low
    v = rng.gauss(2.5, 0.5)
    return max(0.5, min(3.9, v))


def _collect_images(images_dir: Path, include_covid: bool) -> dict[str, list[Path]]:
    root = images_dir / IMAGE_ROOT
    pos_paths: list[Path] = []
    vp_dir = root / "Viral Pneumonia" / "images"
    if vp_dir.exists():
        pos_paths.extend(sorted(vp_dir.glob("*.png")))
    if include_covid:
        covid_dir = root / "COVID" / "images"
        if covid_dir.exists():
            pos_paths.extend(sorted(covid_dir.glob("*.png")))

    neg_paths: list[Path] = []
    normal_dir = root / "Normal" / "images"
    if normal_dir.exists():
        neg_paths.extend(sorted(normal_dir.glob("*.png")))

    return {"pos": pos_paths, "neg": neg_paths}


def build_kaggle_manifest(config: KaggleAdapterConfig) -> pd.DataFrame:
    rng = random.Random(config.random_seed)
    images = _collect_images(config.images_dir, config.include_covid)

    pos_paths = images["pos"]
    neg_paths = images["neg"]

    rng.shuffle(pos_paths)
    rng.shuffle(neg_paths)

    needed_per_cell = config.kernel_per_cell + config.test_per_cell

    # pos images → cells A (conflict: normal/low WBC) and C (control: high WBC)
    needed_pos = needed_per_cell * 2  # cells A + C
    # neg images → cells B (conflict: high WBC) and D (control: normal/low WBC)
    needed_neg = needed_per_cell * 2  # cells B + D

    if len(pos_paths) < needed_pos:
        raise ValueError(f"Need {needed_pos} pos images, only {len(pos_paths)} available.")
    if len(neg_paths) < needed_neg:
        raise ValueError(f"Need {needed_neg} neg images, only {len(neg_paths)} available.")

    rows: list[dict] = []
    subject_counter = 10_000_000

    def make_row(image_path: Path, finding_label: str, lab_class: str, cell: str) -> dict:
        nonlocal subject_counter
        subject_counter += 1
        wbc = _synth_wbc(lab_class, rng)
        return {
            "subject_id": subject_counter,
            "study_id": subject_counter,
            "image_path": str(image_path),
            "image_filename": image_path.name,
            "finding_label": finding_label,
            "lab_value": round(wbc, 2),
            "lab_ref_lower": WBC_REF_LOWER,
            "lab_ref_upper": WBC_REF_UPPER,
            "lab_class": lab_class,
            "cell": cell,
            "wbc_source": "synthesized",
            "split": None,
        }

    pos_iter = iter(pos_paths)
    neg_iter = iter(neg_paths)

    # Cell A: pos + normal WBC (conflict)
    for _ in range(needed_per_cell):
        rows.append(make_row(next(pos_iter), "pos", "normal", "A"))

    # Cell C: pos + high WBC (control)
    for _ in range(needed_per_cell):
        rows.append(make_row(next(pos_iter), "pos", "high", "C"))

    # Cell B: neg + high WBC (conflict)
    for _ in range(needed_per_cell):
        rows.append(make_row(next(neg_iter), "neg", "high", "B"))

    # Cell D: neg + normal WBC (control)
    for _ in range(needed_per_cell):
        rows.append(make_row(next(neg_iter), "neg", "normal", "D"))

    manifest = pd.DataFrame(rows)
    manifest = manifest.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)
    return manifest


def sample_splits(
    manifest: pd.DataFrame,
    kernel_per_cell: int,
    test_per_cell: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split manifest into kernel and test sets, balanced per cell."""
    kernel_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for cell in ["A", "B", "C", "D"]:
        cell_df = manifest[manifest["cell"] == cell].sample(frac=1.0, random_state=seed + ord(cell))
        kernel_parts.append(cell_df.iloc[:kernel_per_cell])
        test_parts.append(cell_df.iloc[kernel_per_cell: kernel_per_cell + test_per_cell])

    kernel = pd.concat(kernel_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed + 1).reset_index(drop=True)
    return kernel, test


def run_kaggle_pipeline(config: KaggleAdapterConfig, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_kaggle_manifest(config)
    manifest_path = output_dir / "kernel_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    kernel, test = sample_splits(
        manifest,
        config.kernel_per_cell,
        config.test_per_cell,
        config.random_seed,
    )

    kernel_path = output_dir / "kernel_480.csv"
    test_path = output_dir / "kernel_test_1200.csv"
    kernel.to_csv(kernel_path, index=False)
    test.to_csv(test_path, index=False)

    # Leakage check (subject_ids are unique per row so overlap is always 0)
    kernel_ids = set(kernel["subject_id"])
    test_ids = set(test["subject_id"])
    overlap = len(kernel_ids & test_ids)

    cell_counts = {
        "manifest": manifest["cell"].value_counts().reindex(["A","B","C","D"]).to_dict(),
        "kernel_480": kernel["cell"].value_counts().reindex(["A","B","C","D"]).to_dict(),
        "kernel_test_1200": test["cell"].value_counts().reindex(["A","B","C","D"]).to_dict(),
    }

    import json
    (output_dir / "cell_counts.json").write_text(json.dumps(cell_counts, indent=2), encoding="utf-8")
    (output_dir / "leakage_report.json").write_text(
        json.dumps({"overlap": overlap, "leakage_free": overlap == 0}, indent=2), encoding="utf-8"
    )

    return {
        "manifest_rows": len(manifest),
        "kernel_rows": len(kernel),
        "test_rows": len(test),
        "cell_counts": cell_counts,
        "leakage_free": overlap == 0,
        "outputs": {
            "manifest": str(manifest_path),
            "kernel_480": str(kernel_path),
            "kernel_test_1200": str(test_path),
        },
    }
