"""Train and evaluate the male/female voice model (the 'Identify Unknown Voice' feature).

Protocol: the same 64 development / 17 test people as the screening model. Person-wise 5-fold cross-validation
(3 repeats) on the development people, training on clean plus simulated-microphone windows and scoring clean and
simulated-microphone audio of the held-out people (median over windows). The test people are scored once.

    python scripts/train_voice_id.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config, voice_id  # noqa: E402

MIC = ["mic1", "mic2", "mic3"]


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    matrix = voice_id.feature_matrix(frame.to_dict("records"))
    return pd.DataFrame(matrix, columns=voice_id.INPUT_FEATURES, index=frame.index)


def build(kind: str) -> Pipeline:
    """'pitch': log2 F0 only. 'spectral': pitch plus 13 MFCC means (strongly regularised)."""
    columns = [0] if kind == "pitch" else list(range(len(voice_id.INPUT_FEATURES)))
    return Pipeline([("select", ColumnTransformer([("keep", "passthrough", columns)])),
                     ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=1.0 if kind == "pitch" else 0.05, class_weight="balanced",
                                                max_iter=3000))])


def fit(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, kinds=("pitch", "spectral")) -> dict:
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=config.SEED).split(X, y, groups))
    return {k: CalibratedClassifierCV(build(k), method="sigmoid", cv=splits).fit(X.to_numpy(), y) for k in kinds}


def combine(models: dict, kind: str) -> "voice_id.AveragedModel":
    return voice_id.AveragedModel(list(models.values()) if kind == "ensemble" else [models[kind]])


def per_person(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    return frame[["subject_id", "version", "female"]].assign(p=p).groupby(["subject_id", "version"]).agg(
        p=("p", "median"), female=("female", "first")).reset_index()


def metrics(agg: pd.DataFrame) -> dict:
    low, high = voice_id.UNCERTAIN_BAND
    y, p = agg.female.to_numpy(), agg.p.to_numpy()
    confident = (p <= low) | (p >= high)
    return {"people_x_versions": int(len(agg)), "accuracy": float(((p >= 0.5) == y).mean()), "auc": float(roc_auc_score(y, p)),
            "female_recall": float((p[y == 1] >= 0.5).mean()), "male_recall": float((p[y == 0] < 0.5).mean()),
            "share_called_uncertain": float(1 - confident.mean()),
            "accuracy_when_confident": float(((p[confident] >= 0.5) == y[confident]).mean()) if confident.any() else None}


def main() -> None:
    windows = pd.read_csv(config.DATA_PROCESSED / "windows_features.csv")
    sex = pd.read_csv(config.DATA_PROCESSED / "figshare_features.csv")[["subject_id", "sex"]]
    windows = windows.merge(sex, on="subject_id")
    windows["female"] = (windows.sex == "F").astype(int)
    windows = windows[windows.version.isin(["clean", *MIC])].reset_index(drop=True)  # pitch-shifted copies would change the label's meaning
    X = prepare(windows)
    dev, test = (~windows.test).to_numpy(), windows.test.to_numpy()
    persons = windows[dev].drop_duplicates("subject_id")[["subject_id", "female"]].reset_index(drop=True)

    kinds = ["pitch", "spectral", "ensemble"]
    oof = {k: [] for k in kinds}
    for repeat in range(3):
        folds = StratifiedKFold(5, shuffle=True, random_state=config.SEED + repeat).split(persons, persons.female)
        for train_idx, held_idx in folds:
            in_train = windows.subject_id.isin(persons.subject_id[train_idx]).to_numpy()
            held = windows.subject_id.isin(persons.subject_id[held_idx]).to_numpy()
            models = fit(X[in_train], windows.female[in_train].to_numpy(), windows.subject_id[in_train].to_numpy())
            for k in kinds:
                oof[k].append(per_person(windows[held], combine(models, k).predict_proba(X[held].to_numpy())[:, 1]))
    versions_of = [("clean", ["clean"]), ("simulated microphone", MIC)]
    cv = {k: {name: metrics(pd.concat(oof[k])[lambda d, v=versions: d.version.isin(v)]) for name, versions in versions_of}
          for k in kinds}
    for k in kinds:
        print(f"CV {k:9s} clean acc {cv[k]['clean']['accuracy']:.3f} | mic acc {cv[k]['simulated microphone']['accuracy']:.3f}"
              f" | uncertain {cv[k]['clean']['share_called_uncertain']:.0%} | accuracy when confident "
              f"{cv[k]['clean']['accuracy_when_confident']:.3f}")

    final_models = fit(X[dev], windows.female[dev].to_numpy(), windows.subject_id[dev].to_numpy())
    test_report = {}
    for k in kinds:
        agg_test = per_person(windows[test], combine(final_models, k).predict_proba(X[test].to_numpy())[:, 1])
        test_report[k] = {name: metrics(agg_test[agg_test.version.isin(versions)]) for name, versions in versions_of}
    test_people = per_person(windows[test], combine(final_models, "ensemble").predict_proba(X[test].to_numpy())[:, 1])
    test_people = test_people[test_people.version == "clean"]
    rng = np.random.default_rng(0)
    boot = [((test_people.p.to_numpy()[i] >= 0.5) == test_people.female.to_numpy()[i]).mean()
            for i in (rng.integers(0, len(test_people), len(test_people)) for _ in range(2000))]
    test_report["ensemble"]["clean"]["accuracy_ci95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    config.MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": combine(final_models, "ensemble"), "features": voice_id.INPUT_FEATURES,
                 "uncertain_band": voice_id.UNCERTAIN_BAND, "classes": {"0": "male", "1": "female"}},
                config.MODELS / "voice_id.joblib")
    report = {"task": "male/female voice from a sustained /a/", "features": voice_id.INPUT_FEATURES,
              "model": "average of two calibrated logistic regressions: pitch only, and pitch + 13 MFCC means",
              "training": "64 development people, clean + 3 simulated-microphone versions, 1.5 s windows, median per person",
              "uncertain_band": list(voice_id.UNCERTAIN_BAND), "deployed": "ensemble", "cv": cv, "test": test_report,
              "people": {"development_female": int(persons.female.sum()), "development_male": int((1 - persons.female).sum())}}
    (config.REPORTS / "voice_id.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Test (17 people), ensemble:", json.dumps(test_report["ensemble"], indent=1))


if __name__ == "__main__":
    main()
