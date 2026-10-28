"""One-time fix: make models/feature_model.joblib loadable without pandas.

The deployed feature-model bundle was saved with a small pandas DataFrame (the SHAP background sample). That makes the
whole file unloadable on machines where Windows Application Control blocks pandas. This script reads the bundle with
every pandas object replaced by an inert stub, rebuilds the background sample as a plain numpy array from the
training windows (data/processed/windows_features.csv, read with the csv module), and saves the bundle again. The
original file is kept as feature_model.joblib.bak.

    python scripts/migrate_bundle_to_numpy.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
from joblib import numpy_pickle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config  # noqa: E402


class _Stub:
    """Stands in for any pandas class or function while unpickling."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return _Stub()

    def __setstate__(self, state):
        pass


class _PandasFreeUnpickler(numpy_pickle.NumpyUnpickler):
    def find_class(self, module, name):
        if module.split(".")[0] == "pandas":
            return _Stub
        return super().find_class(module, name)


def load_without_pandas(path: Path):
    original = numpy_pickle.NumpyUnpickler
    numpy_pickle.NumpyUnpickler = _PandasFreeUnpickler
    try:
        return joblib.load(path)
    finally:
        numpy_pickle.NumpyUnpickler = original


def main() -> None:
    path = config.MODELS / "feature_model.joblib"
    bundle = load_without_pandas(path)
    if isinstance(bundle.get("shap_background"), np.ndarray):
        print("Already numpy-only; nothing to do.")
        return
    names = bundle["features"]
    with open(config.DATA_PROCESSED / "windows_features.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["test"] == "False" and r["version"] == "clean"]
    rng = np.random.default_rng(config.SEED)
    pick = rng.choice(len(rows), size=min(20, len(rows)), replace=False)
    background = np.array([[float(rows[i][n]) if rows[i][n] not in ("", "nan") else np.nan for n in names] for i in pick])
    bundle["shap_background"] = background
    bundle["shap_background_columns"] = list(names)
    shutil.copy2(path, path.with_suffix(".joblib.bak"))
    joblib.dump(bundle, path)
    print(f"Rewrote {path.name}: shap_background is now a numpy array of shape {background.shape}; "
          f"backup kept as {path.name}.bak")


if __name__ == "__main__":
    main()
