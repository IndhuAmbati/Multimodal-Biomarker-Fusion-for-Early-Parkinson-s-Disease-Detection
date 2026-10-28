"""Build the window-level feature table: every person x every perturbation version x every 1.5 s window.

    python scripts/prepare_windows.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

SRC = str(Path(__file__).resolve().parents[1] / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = SRC + os.pathsep + os.environ.get("PYTHONPATH", "")

from pdscreen import augment, config  # noqa: E402


def windows_for(subject_id: str, label: int, test: bool, age: float, path: str, version: str) -> list[dict]:
    from pdscreen import audio, features

    raw, rate = audio.load_audio(path)
    x, r = augment.make_version(raw, rate, version, subject_id)
    try:
        y, _ = audio.preprocess(x, r)
    except audio.AudioError:
        return []
    rows = []
    for i, window in enumerate(audio.analysis_windows(y)):
        rows.append({"subject_id": subject_id, "label": label, "test": test, "age": age, "version": version,
                     "family": augment.family(version), "window": i, **features.extract_features(window)})
    return rows


def main() -> None:
    table = pd.read_csv(config.DATA_PROCESSED / "figshare_features.csv")
    jobs = [(r.subject_id, int(r.label), bool(r.test), float(r.age), r.path, v)
            for r in table.itertuples() for v in augment.VERSIONS]
    start = time.perf_counter()
    results = Parallel(n_jobs=3, batch_size=4)(delayed(windows_for)(*job) for job in jobs)
    frame = pd.DataFrame([row for rows in results for row in rows])
    frame.to_csv(config.DATA_PROCESSED / "windows_features.csv", index=False)
    print(f"{len(jobs)} person-versions -> {len(frame)} windows in {time.perf_counter() - start:.0f} s")
    print(frame.groupby(["family"]).size().to_dict(), "| persons", frame.subject_id.nunique())


if __name__ == "__main__":
    main()
