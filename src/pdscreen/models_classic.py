"""Feature-based classifiers and nested, subject-grouped cross-validation."""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config, features

FEATURE_SETS = {
    "clinical": (features.F0_FEATURES + features.JITTER_FEATURES + features.SHIMMER_FEATURES
                 + features.NOISE_FEATURES + features.NONLINEAR_FEATURES + features.SPECTRAL_FEATURES),
    "all": features.FEATURE_NAMES,
}


def model_specs(seed: int = config.SEED) -> dict[str, tuple[Pipeline, dict]]:
    """Each model is a pipeline (impute -> scale -> classifier) plus a small grid.

    Grids are deliberately small: with 64 development subjects a large search
    would overfit the inner cross-validation.
    """
    impute = ("impute", SimpleImputer(strategy="median"))
    return {
        "Logistic regression": (
            Pipeline([impute, ("scale", StandardScaler()),
                      ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            {"clf__C": [0.01, 0.1, 1.0]}),
        "SVM (RBF)": (
            Pipeline([impute, ("scale", StandardScaler()),
                      ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
            {"clf__C": [0.3, 1.0, 3.0, 10.0], "clf__gamma": ["scale", 0.003, 0.01, 0.03]}),
        "Random forest": (
            Pipeline([impute, ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                             max_features="sqrt", n_jobs=1, random_state=seed))]),
            {"clf__max_depth": [3, None], "clf__min_samples_leaf": [1, 3]}),
        "XGBoost": (
            Pipeline([impute, ("clf", XGBClassifier(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                                    eval_metric="logloss", n_jobs=1, random_state=seed))]),
            {"clf__max_depth": [2, 3], "clf__n_estimators": [150, 400]}),
    }


def fit_calibrated(pipeline: Pipeline, grid: dict, X, y, groups, seed: int = config.SEED,
                   inner_folds: int = 4) -> tuple[CalibratedClassifierCV, dict]:
    """Tune on grouped inner folds, then wrap the best model in sigmoid calibration.

    Class imbalance is handled inside each model (class weights for LR, SVM
    and RF; ``scale_pos_weight`` for XGBoost) using training-fold counts only.
    """
    pipeline = clone(pipeline)
    if isinstance(pipeline.named_steps["clf"], XGBClassifier):
        positives = max(1, int(np.sum(y == 1)))
        pipeline.set_params(clf__scale_pos_weight=float(np.sum(y == 0)) / positives)
    search = GridSearchCV(pipeline, grid, scoring="roc_auc",
                          cv=StratifiedGroupKFold(inner_folds, shuffle=True, random_state=seed), n_jobs=1)
    search.fit(X, y, groups=groups)
    calibrated = CalibratedClassifierCV(clone(search.best_estimator_), method="sigmoid",
                                        cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed))
    calibrated.fit(X, y)
    return calibrated, {k.replace("clf__", ""): v for k, v in search.best_params_.items()}


def nested_cv(pipeline: Pipeline, grid: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              folds: int = 5, repeats: int = 3, seed: int = config.SEED):
    """Repeated, stratified, subject-grouped outer CV with inner tuning.

    Returns per-fold metrics and an out-of-fold probability per subject,
    averaged over the repeats.
    """
    from .evaluation import classification_metrics

    oof = np.zeros((repeats, len(y)))
    fold_metrics, chosen = [], []
    for r in range(repeats):
        outer = StratifiedGroupKFold(folds, shuffle=True, random_state=seed + r)
        for train, test in outer.split(X, y, groups):
            model, params = fit_calibrated(pipeline, grid, X[train], y[train], groups[train], seed=seed + r)
            p = model.predict_proba(X[test])[:, 1]
            oof[r, test] = p
            fold_metrics.append(classification_metrics(y[test], p))
            chosen.append(params)
    return fold_metrics, oof.mean(axis=0), chosen
