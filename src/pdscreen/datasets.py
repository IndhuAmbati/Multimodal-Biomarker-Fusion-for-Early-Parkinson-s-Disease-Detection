"""Dataset manifests with one row per recording and an explicit subject ID."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def figshare_manifest(root: Path = config.FIGSHARE_DIR) -> pd.DataFrame:
    """Iyer et al. (2023) sustained /a/ recordings: 40 PD, 41 healthy controls.

    Each participant contributed one recording, so the sample ID is the
    subject ID. Age and sex come from the dataset's demographics sheet.
    """
    demographics = pd.read_excel(root / "Demographics_age_sex.xlsx", sheet_name="Parselmouth")
    demographics = demographics.rename(columns={"Sample ID": "subject_id", "Label": "group", "Age": "age", "Sex": "sex"})
    rows = []
    for folder, label in [("PD_AH", 1), ("HC_AH", 0)]:
        for path in sorted((root / folder).glob("*.wav")):
            rows.append({"subject_id": path.stem, "label": label, "path": str(path)})
    manifest = pd.DataFrame(rows).merge(demographics[["subject_id", "age", "sex", "group"]], on="subject_id", how="left")
    missing = manifest["age"].isna().sum()
    if missing:
        raise ValueError(f"{missing} recordings have no demographics row.")
    expected = manifest["group"].map({"PwPD": 1, "HC": 0})
    if not (expected == manifest["label"]).all():
        raise ValueError("Folder labels disagree with the demographics sheet.")
    if manifest["subject_id"].duplicated().any():
        raise ValueError("Duplicate subject IDs found.")
    return manifest.drop(columns="group")


def uci_parkinsons(root: Path = config.UCI_DIR) -> pd.DataFrame:
    """Little et al. (2007) UCI Parkinson's features: 195 recordings from 32 people.

    The ``name`` column encodes the subject (``phon_R01_S01_1`` -> ``S01``);
    each person has about six recordings, which is why subject-wise splitting
    matters here.
    """
    frame = pd.read_csv(root / "parkinsons.data")
    frame["subject_id"] = frame["name"].str.extract(r"_(S\d+)_")[0]
    if frame["subject_id"].isna().any():
        raise ValueError("Could not parse subject IDs from the UCI 'name' column.")
    return frame


def uci_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in {"name", "status", "subject_id"}]


def dataset_summary(manifest: pd.DataFrame) -> dict:
    by_label = manifest.groupby("label")
    summary = {
        "subjects": int(manifest["subject_id"].nunique()),
        "recordings": int(len(manifest)),
        "pd_subjects": int(manifest.loc[manifest.label == 1, "subject_id"].nunique()),
        "hc_subjects": int(manifest.loc[manifest.label == 0, "subject_id"].nunique()),
    }
    if "age" in manifest:
        for label, name in [(1, "pd"), (0, "hc")]:
            group = manifest[manifest.label == label]
            summary[f"{name}_age_mean"] = round(float(group.age.mean()), 1)
            summary[f"{name}_age_range"] = [float(group.age.min()), float(group.age.max())]
            summary[f"{name}_female"] = int((group.sex == "F").sum())
            summary[f"{name}_male"] = int((group.sex == "M").sum())
            summary[f"{name}_under_50"] = int((group.age < 50).sum())
    if "duration_seconds" in manifest:
        summary["duration_seconds"] = {k: [round(float(v.min()), 2), round(float(v.median()), 2), round(float(v.max()), 2)]
                                       for k, v in by_label["duration_seconds"]}
    return summary


def holdout_split(manifest: pd.DataFrame, test_fraction: float = 0.2, seed: int = config.SEED) -> np.ndarray:
    """Boolean mask of held-out test subjects, stratified by label and age band.

    Stratifying on age as well keeps the young-control imbalance the same in
    the development and test sets, so the test set is not easier or harder by
    chance.
    """
    rng = np.random.default_rng(seed)
    subjects = manifest.drop_duplicates("subject_id")
    strata = subjects["label"].astype(str) + "_" + (subjects["age"] >= 50).astype(str)
    test_ids: list[str] = []
    for _, group in subjects.groupby(strata):
        ids = group["subject_id"].to_numpy()
        rng.shuffle(ids)
        test_ids.extend(ids[: int(round(len(ids) * test_fraction))])
    return manifest["subject_id"].isin(test_ids).to_numpy()
