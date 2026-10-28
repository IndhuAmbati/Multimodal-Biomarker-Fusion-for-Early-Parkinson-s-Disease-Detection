"""Show how recording-level splitting inflates results on the UCI Parkinson's data.

The UCI set has ~6 recordings per person. Splitting by recording puts the
same person in training and test folds; splitting by subject does not. Both
use the same models and the same stratified 5-fold scheme, repeated 5 times.

    python scripts/evaluate_uci.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config, datasets, evaluation  # noqa: E402
from pdscreen.models_classic import model_specs  # noqa: E402

DEFAULTS = {"Logistic regression": {"clf__C": 0.1}, "SVM (RBF)": {"clf__C": 1.0, "clf__gamma": "scale"},
            "Random forest": {"clf__max_depth": None, "clf__min_samples_leaf": 1},
            "XGBoost": {"clf__max_depth": 3, "clf__n_estimators": 150}}


def subject_level(y, p, groups):
    """Average recording probabilities per subject for a subject-level score."""
    ids = np.unique(groups)
    return (np.array([y[groups == g][0] for g in ids]), np.array([p[groups == g].mean() for g in ids]))


def run(splitter_factory, X, y, groups, pipeline, repeats=5):
    aucs, accs, subject_aucs = [], [], []
    for r in range(repeats):
        splitter = splitter_factory(r)
        p = np.zeros(len(y))
        for train, test in splitter.split(X, y, groups):
            model = clone(pipeline)
            if "XGB" in type(model.named_steps["clf"]).__name__:
                model.set_params(clf__scale_pos_weight=float((y[train] == 0).sum()) / (y[train] == 1).sum())
            model.fit(X[train], y[train])
            p[test] = model.predict_proba(X[test])[:, 1]
        m = evaluation.classification_metrics(y, p)
        aucs.append(m["roc_auc"])
        accs.append(m["accuracy"])
        ys, ps = subject_level(y, p, groups)
        subject_aucs.append(evaluation.classification_metrics(ys, ps)["roc_auc"])
    return {"recording_auc": float(np.mean(aucs)), "recording_accuracy": float(np.mean(accs)),
            "subject_auc": float(np.mean(subject_aucs))}


def main() -> None:
    frame = datasets.uci_parkinsons()
    columns = datasets.uci_feature_columns(frame)
    X, y, groups = frame[columns].to_numpy(), frame.status.to_numpy(), frame.subject_id.to_numpy()
    results = {}
    for name, (pipeline, _) in model_specs().items():
        pipeline = clone(pipeline).set_params(**DEFAULTS[name])
        if name == "SVM (RBF)":
            pipeline.set_params(clf__probability=True)
        by_recording = run(lambda r: StratifiedKFold(5, shuffle=True, random_state=r), X, y, groups, pipeline)
        by_subject = run(lambda r: StratifiedGroupKFold(5, shuffle=True, random_state=r), X, y, groups, pipeline)
        results[name] = {"split_by_recording": by_recording, "split_by_subject": by_subject}
        print(f"{name:20s} accuracy by recording {by_recording['recording_accuracy']:.3f} -> by subject "
              f"{by_subject['recording_accuracy']:.3f} | AUC {by_recording['recording_auc']:.3f} -> "
              f"{by_subject['recording_auc']:.3f}")
    summary = {"dataset": "UCI Parkinson's (Little et al. 2007): 195 recordings, 32 subjects (24 PD, 8 healthy)",
               "protocol": "stratified 5-fold, repeated 5 times; same models, only the split changes",
               "results": results}
    (config.REPORTS / "uci_leakage_demo.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
