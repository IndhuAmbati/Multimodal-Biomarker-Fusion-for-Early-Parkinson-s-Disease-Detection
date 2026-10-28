"""Train and evaluate the feature-based baselines (LR, SVM, RF, XGBoost).

Protocol
--------
1. 17 subjects are held out as a test set (stratified by label and age band)
   and are not touched until the very end.
2. On the 64 development subjects: 3 x 5-fold stratified, subject-grouped
   outer CV. Inside each outer training fold, hyperparameters are tuned with
   4-fold grouped CV and the chosen model is sigmoid-calibrated.
3. The configuration with the best mean outer-CV ROC-AUC is refitted on all
   development subjects and evaluated once on the held-out test set.
4. Confound checks: an age-only model, and the best model's performance in
   subjects aged 50 or over.

    python scripts/train_classic.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config, evaluation, explain, features  # noqa: E402
from pdscreen.models_classic import FEATURE_SETS, fit_calibrated, model_specs, nested_cv  # noqa: E402


def main(repeats: int = 3) -> None:
    config.MODELS.mkdir(parents=True, exist_ok=True)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(config.DATA_PROCESSED / "figshare_features.csv")
    dev, test = table[~table.test].reset_index(drop=True), table[table.test].reset_index(drop=True)
    y_dev, g_dev = dev.label.to_numpy(), dev.subject_id.to_numpy()

    results, oof = {}, {}
    for set_name, columns in FEATURE_SETS.items():
        for model_name, (pipeline, grid) in model_specs().items():
            key = f"{model_name} | {set_name} features"
            start = time.perf_counter()
            folds, p, chosen = nested_cv(pipeline, grid, dev[columns].to_numpy(), y_dev, g_dev, repeats=repeats)
            results[key] = {"model": model_name, "feature_set": set_name, "n_features": len(columns),
                            "cv_folds": evaluation.summarise_folds(folds),
                            "cv_pooled_oof": evaluation.classification_metrics(y_dev, p),
                            "chosen_params": chosen, "seconds": round(time.perf_counter() - start, 1)}
            oof[key] = p
            print(f"{key:45s} CV AUC {results[key]['cv_folds']['roc_auc']['mean']:.3f} "
                  f"+/- {results[key]['cv_folds']['roc_auc']['std']:.3f}  ({results[key]['seconds']} s)")

    # Confound check 1: how well does age alone separate the groups?
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    age_folds, age_p, _ = nested_cv(Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())]),
                                    {"clf__C": [1.0]}, dev[["age"]].to_numpy(), y_dev, g_dev, repeats=repeats)
    results["Age only (confound check)"] = {"model": "Logistic regression on age", "feature_set": "age",
                                            "cv_folds": evaluation.summarise_folds(age_folds),
                                            "cv_pooled_oof": evaluation.classification_metrics(y_dev, age_p)}
    print(f"{'Age only (confound check)':45s} CV AUC {results['Age only (confound check)']['cv_folds']['roc_auc']['mean']:.3f}")

    best_key = max((k for k in results if k in oof), key=lambda k: results[k]["cv_folds"]["roc_auc"]["mean"])
    best = results[best_key]
    columns = FEATURE_SETS[best["feature_set"]]
    pipeline, grid = model_specs()[best["model"]]
    print(f"\nSelected: {best_key}")

    # Confound check 2: best model's out-of-fold performance in subjects aged 50+.
    older = dev.age.to_numpy() >= 50
    best["cv_oof_age_50_plus"] = evaluation.classification_metrics(y_dev[older], oof[best_key][older])
    best["cv_oof_age_50_plus"]["subjects"] = {"pd": int(y_dev[older].sum()), "hc": int((1 - y_dev[older]).sum())}
    best["confidence_band_accuracy_oof"] = evaluation.confidence_band_accuracy(y_dev, oof[best_key])

    # Final fit on all development subjects, single evaluation on the test set.
    final, params = fit_calibrated(pipeline, grid, dev[columns].to_numpy(), y_dev, g_dev)
    p_test = final.predict_proba(test[columns].to_numpy())[:, 1]
    test_metrics = evaluation.classification_metrics(test.label.to_numpy(), p_test)
    test_metrics["ci95"] = evaluation.bootstrap_ci(test.label.to_numpy(), p_test)
    best["test"] = test_metrics
    best["final_params"] = params
    print("Held-out test:", {k: round(v, 3) for k, v in test_metrics.items() if isinstance(v, float)})

    # Explainability: SHAP values of the calibrated probability.
    background = dev[columns]
    shap_summary = explain.global_shap(final, background, columns, config.FIGURES / "shap_summary.png")
    best["shap_mean_abs_top10"] = shap_summary

    bundle = {"model": final, "features": columns, "name": best_key, "params": params,
              "background": background.median().to_dict(),
              "training_ranges": {c: [float(dev[c].quantile(0.01)), float(dev[c].quantile(0.99))] for c in columns},
              "shap_background": background.sample(min(20, len(background)), random_state=config.SEED)}
    joblib.dump(bundle, config.MODELS / "feature_model_plain.joblib")  # first-generation whole-recording model; benchmark only

    # Figures from out-of-fold predictions (development) and the test set.
    ranked = sorted((k for k in oof), key=lambda k: -results[k]["cv_folds"]["roc_auc"]["mean"])
    curves = {k.replace(" features", ""): (y_dev, oof[k]) for k in ranked[:4]}
    curves["Age only"] = (y_dev, age_p)
    evaluation.plot_roc(curves, config.FIGURES / "roc_cv_feature_models.png", "Out-of-fold ROC (64 development subjects)")
    evaluation.plot_calibration({best_key.split(" |")[0]: (y_dev, oof[best_key])},
                                config.FIGURES / "calibration_feature_model.png", "Calibration (out-of-fold)")
    evaluation.plot_confusion(best["cv_pooled_oof"]["confusion_matrix"], f"{best['model']}: out-of-fold",
                              config.FIGURES / "confusion_feature_model_cv.png")
    evaluation.plot_confusion(test_metrics["confusion_matrix"], f"{best['model']}: held-out test (17)",
                              config.FIGURES / "confusion_feature_model_test.png")
    pd.DataFrame({"subject_id": dev.subject_id, "label": y_dev, "age": dev.age, **{k: v for k, v in oof.items()}}) \
        .to_csv(config.REPORTS / "oof_feature_models.csv", index=False)
    pd.DataFrame({"subject_id": test.subject_id, "label": test.label, "p_pd": p_test}) \
        .to_csv(config.REPORTS / "test_predictions_feature_model.csv", index=False)

    (config.REPORTS / "feature_models.json").write_text(json.dumps({"selected": best_key, "results": results},
                                                                  indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
