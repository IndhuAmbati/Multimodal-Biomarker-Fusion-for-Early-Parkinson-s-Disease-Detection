"""Build the subject manifest, the held-out test split and the feature table.

Run once after the raw datasets are in data/raw:

    python scripts/prepare_data.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import audio, config, datasets, features  # noqa: E402


def main() -> None:
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    config.REPORTS.mkdir(parents=True, exist_ok=True)
    manifest = datasets.figshare_manifest()
    manifest["test"] = datasets.holdout_split(manifest)

    rows, signals = [], {}
    start = time.perf_counter()
    for record in manifest.itertuples():
        raw, rate = audio.load_audio(record.path)
        cleaned, quality = audio.preprocess(raw, rate)
        signals[record.subject_id] = cleaned
        rows.append({"subject_id": record.subject_id, "duration_seconds": quality.duration_seconds,
                     "voiced_seconds": quality.voiced_seconds, **features.extract_features(cleaned)})
    table = manifest.merge(pd.DataFrame(rows), on="subject_id")
    table.to_csv(config.DATA_PROCESSED / "figshare_features.csv", index=False)
    np.savez_compressed(config.DATA_PROCESSED / "figshare_signals.npz", **signals)

    summary = {
        "dataset": "Iyer et al. 2023, Voice Samples for Patients with Parkinson's Disease and Healthy Controls "
                   "(figshare, doi:10.6084/m9.figshare.23849127, CC BY 4.0)",
        "task": "sustained vowel /a/, recorded over the telephone at 8 kHz",
        "all": datasets.dataset_summary(table),
        "development": datasets.dataset_summary(table[~table.test]),
        "test": datasets.dataset_summary(table[table.test]),
        "missing_feature_values": {k: int(v) for k, v in table[features.FEATURE_NAMES].isna().sum().items() if v},
        "uci": {},
        "extraction_seconds": round(time.perf_counter() - start, 1),
    }
    uci = datasets.uci_parkinsons()
    summary["uci"] = {
        "dataset": "Little et al. 2007, UCI Parkinson's (id 174)",
        "recordings": int(len(uci)),
        "subjects": int(uci.subject_id.nunique()),
        "pd_subjects": int(uci.loc[uci.status == 1, "subject_id"].nunique()),
        "hc_subjects": int(uci.loc[uci.status == 0, "subject_id"].nunique()),
        "recordings_per_subject": [int(uci.groupby("subject_id").size().min()), int(uci.groupby("subject_id").size().max())],
    }
    (config.REPORTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
