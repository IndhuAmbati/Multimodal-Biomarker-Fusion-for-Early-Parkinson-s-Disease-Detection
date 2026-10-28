"""Train the feature model to be robust to pitch changes, microphones and recording length.

Differences from scripts/train_classic.py:
- Features come from fixed 1.5 s windows; a person's score is the median over their windows.
- Absolute-pitch features (mean/min/max F0, absolute jitter) are optional, so the model need not depend on
  whether someone says a higher or lower 'aaah'.
- Training rows include pitch-shifted (formant-preserving) and microphone-simulated copies of each training
  person, kept in the same fold as the original person.
- Selection looks at clean AND perturbed performance on out-of-fold people, and the decision threshold is
  chosen on out-of-fold data to keep false alarms on healthy people low.
- Test people (17) are used once, at the end.

    python scripts/train_robust.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config, evaluation, explain, features  # noqa: E402

ABSOLUTE_PITCH = ["f0_mean_hz", "f0_min_hz", "f0_max_hz", "jitter_abs"]
CLINICAL = (features.F0_FEATURES + features.JITTER_FEATURES + features.SHIMMER_FEATURES + features.NOISE_FEATURES
            + features.NONLINEAR_FEATURES + features.SPECTRAL_FEATURES)
INVARIANT = [f for f in features.FEATURE_NAMES if f not in ABSOLUTE_PITCH]
FEATURE_SETS = {
    "all 50": features.FEATURE_NAMES,
    "pitch-invariant 46": INVARIANT,
    "pitch-invariant, no MFCC std 33": [f for f in INVARIANT if not (f.startswith("mfcc") and f.endswith("_std"))],
    "clinical, pitch-invariant 20": [f for f in CLINICAL if f not in ABSOLUTE_PITCH],
}


def make_model(kind: str, seed: int = config.SEED) -> Pipeline:
    impute = ("impute", SimpleImputer(strategy="median"))
    if kind == "LR":
        return Pipeline([impute, ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=0.05, class_weight="balanced", max_iter=3000))])
    return Pipeline([impute, ("clf", RandomForestClassifier(
        n_estimators=150, max_depth=6, min_samples_leaf=5, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=2, random_state=seed))])


def fit_calibrated(kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> CalibratedClassifierCV:
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=config.SEED).split(X, y, groups))
    return CalibratedClassifierCV(make_model(kind), method="sigmoid", cv=splits).fit(X, y)


def per_person(frame: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Median window probability for each (person, perturbation version)."""
    scored = frame[["subject_id", "version", "family", "label"]].assign(p=p)
    return scored.groupby(["subject_id", "version"], as_index=False).agg(
        label=("label", "first"), family=("family", "first"), p=("p", "median"))


def summarise(agg: pd.DataFrame, threshold: float = 0.5) -> dict:
    out = {}
    for name, sub in [("clean", agg[agg.family == "clean"]), ("perturbed", agg[agg.family != "clean"])]:
        y, p = sub.label.to_numpy(), sub.p.to_numpy()
        sensitivity, specificity = float((p[y == 1] >= threshold).mean()), float((p[y == 0] < threshold).mean())
        out[name] = {"auc": float(roc_auc_score(y, p)), "sensitivity": sensitivity, "specificity": specificity,
                     "balanced_accuracy": (sensitivity + specificity) / 2,
                     "mean_p_healthy": float(p[y == 0].mean()), "mean_p_pd": float(p[y == 1].mean())}
    return out


def choose_threshold(agg: pd.DataFrame, target_specificity: float = 0.80) -> tuple[float, str]:
    """Highest sensitivity that keeps healthy specificity >= target on out-of-fold clean+perturbed people."""
    y, p = agg.label.to_numpy(), agg.p.to_numpy()
    best = None
    for t in np.arange(0.30, 0.86, 0.01):
        sensitivity, specificity = (p[y == 1] >= t).mean(), (p[y == 0] < t).mean()
        if specificity >= target_specificity and (best is None or sensitivity > best[1]):
            best = (float(t), float(sensitivity))
    if best is not None:
        return round(best[0], 2), f"lowest threshold keeping out-of-fold specificity >= {target_specificity:.0%}"
    grid = np.arange(0.30, 0.86, 0.01)
    scores = [((p[y == 1] >= t).mean() + (p[y == 0] < t).mean()) / 2 for t in grid]
    return round(float(grid[int(np.argmax(scores))]), 2), "maximum balanced accuracy (target specificity unreachable)"


def main() -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(config.DATA_PROCESSED / "windows_features.csv")
    dev, test = table[~table.test].reset_index(drop=True), table[table.test].reset_index(drop=True)
    persons = dev.drop_duplicates("subject_id")[["subject_id", "label"]].reset_index(drop=True)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=config.SEED).split(
        persons, persons.label, persons.subject_id))

    results, oof_store = {}, {}
    for set_name, columns in FEATURE_SETS.items():
        for kind in ("LR", "RF"):
            for mode in ("plain", "augmented"):
                key = f"{kind} | {set_name} | {mode}"
                start = time.perf_counter()
                p = np.zeros(len(dev))
                for train_idx, test_idx in folds:
                    train_ids, test_ids = persons.subject_id[train_idx], persons.subject_id[test_idx]
                    train = dev[dev.subject_id.isin(train_ids) & ((dev.version == "clean") | (mode == "augmented"))]
                    held = dev.subject_id.isin(test_ids)
                    model = fit_calibrated(kind, train[columns].to_numpy(), train.label.to_numpy(),
                                           train.subject_id.to_numpy())
                    p[held.to_numpy()] = model.predict_proba(dev.loc[held, columns].to_numpy())[:, 1]
                agg = per_person(dev, p)
                results[key] = {"kind": kind, "feature_set": set_name, "mode": mode, "cv": summarise(agg),
                                "seconds": round(time.perf_counter() - start, 1)}
                oof_store[key] = agg
                c = results[key]["cv"]
                print(f"{key:55s} clean AUC {c['clean']['auc']:.3f} | perturbed AUC {c['perturbed']['auc']:.3f} | "
                      f"perturbed spec {c['perturbed']['specificity']:.2f} sens {c['perturbed']['sensitivity']:.2f}",
                      flush=True)

    def score(key: str) -> float:
        c = results[key]["cv"]
        return 0.5 * (c["clean"]["auc"] + c["perturbed"]["auc"])

    best_key = max(results, key=score)
    best = results[best_key]
    threshold, rule = choose_threshold(oof_store[best_key])
    best["cv_at_threshold"] = summarise(oof_store[best_key], threshold)
    ages = dev.drop_duplicates("subject_id").set_index("subject_id")["age"]
    saved = oof_store[best_key].assign(age=lambda d: d.subject_id.map(ages))
    saved.to_csv(config.REPORTS / "oof_robust_model.csv", index=False)
    older = saved[(saved.family == "clean") & (saved.age >= 50)]
    best["cv_age_50_plus_clean"] = {
        "auc": float(roc_auc_score(older.label, older.p)), "pd": int(older.label.sum()),
        "hc": int((1 - older.label).sum()), "sensitivity": float((older.p[older.label == 1] >= threshold).mean()),
        "specificity": float((older.p[older.label == 0] < threshold).mean())}
    print(f"\nSelected: {best_key} (mean of clean and perturbed CV AUC {score(best_key):.3f}); "
          f"threshold {threshold} ({rule})")

    # Final model on all development people, evaluated once on the held-out test people.
    columns = FEATURE_SETS[best["feature_set"]]
    train = dev if best["mode"] == "augmented" else dev[dev.version == "clean"]
    final = fit_calibrated(best["kind"], train[columns].to_numpy(), train.label.to_numpy(), train.subject_id.to_numpy())
    p_test = final.predict_proba(test[columns].to_numpy())[:, 1]
    agg_test = per_person(test, p_test)
    best["test_at_threshold"] = summarise(agg_test, threshold)
    clean_test = agg_test[agg_test.family == "clean"]
    clean_dev = oof_store[best_key][oof_store[best_key].family == "clean"]
    best["test_clean_metrics"] = {**evaluation.classification_metrics(clean_test.label, clean_test.p, threshold),
                                  "ci95": evaluation.bootstrap_ci(clean_test.label, clean_test.p, threshold)}
    best["cv_clean_metrics"] = evaluation.classification_metrics(clean_dev.label, clean_dev.p, threshold)
    print("Held-out test (17 people), at threshold:", json.dumps(best["test_at_threshold"], default=float)[:600])

    # Out-of-fold curves for the report and a SHAP summary for the new model.
    pert = oof_store[best_key][oof_store[best_key].family != "clean"]
    evaluation.plot_roc({"Clean recordings": (clean_dev.label, clean_dev.p), "Pitch / microphone variants": (pert.label, pert.p)},
                        config.FIGURES / "roc_robust_model.png", f"Out-of-fold ROC: {best['kind']}, {best['feature_set']}")
    evaluation.plot_confusion(best["test_clean_metrics"]["confusion_matrix"], "Deployed model: held-out test (17)",
                              config.FIGURES / "confusion_deployed_test.png")
    clean_windows = dev[dev.version == "clean"]
    best["shap_mean_abs_top10"] = explain.global_shap(final, clean_windows[columns].sample(60, random_state=config.SEED),
                                                       columns, config.FIGURES / "shap_summary.png")

    bundle = {"model": final, "features": columns, "name": best_key, "params": {"kind": best["kind"], "mode": best["mode"]},
              "background": clean_windows[columns].median().to_dict(),
              "training_ranges": {c: [float(clean_windows[c].quantile(0.01)), float(clean_windows[c].quantile(0.99))]
                                  for c in columns},
              "shap_background": clean_windows[columns].sample(20, random_state=config.SEED).to_numpy(),  # numpy, not a DataFrame: keeps the bundle loadable without pandas
              "window_seconds": config.ANALYSIS_WINDOW_SECONDS, "aggregation": "median"}
    joblib.dump(bundle, config.MODELS / "feature_model.joblib")

    cm = best["test_clean_metrics"]
    deployment = {
        "name": f"{best['kind']} on {best['feature_set']} features, {best['mode']} training (1.5 s windows, median)",
        "choice": "robust_feature_model", "components": ["feature_model"], "threshold": threshold,
        "threshold_rule": rule,
        "selection_rule": "highest mean of clean and perturbed out-of-fold ROC-AUC on the 64 development people",
        "evaluation": {"cv": best["cv_clean_metrics"], "test": cm},
        "robustness": {"cv": best["cv_at_threshold"], "test": best["test_at_threshold"]},
        "reference_medians": {"healthy": clean_windows[clean_windows.label == 0][columns].median().round(6).to_dict(),
                              "pd": clean_windows[clean_windows.label == 1][columns].median().round(6).to_dict()},
        "created": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
    }
    old = json.loads((config.MODELS / "deployment.json").read_text(encoding="utf-8"))
    deployment["limitations_markdown"] = old.get("limitations_markdown", "")
    (config.MODELS / "deployment.json").write_text(json.dumps(deployment, indent=2, default=float), encoding="utf-8")
    (config.REPORTS / "robust_feature_model.json").write_text(
        json.dumps({"selected": best_key, "threshold": threshold, "threshold_rule": rule, "results": results},
                   indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
