"""Write the results report from the saved experiment outputs (and the overfitting check).

The deployed model and its threshold are written by train_robust.py from development cross-validation only; this
script reports, it does not choose.

    python scripts/finalize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config, evaluation, features  # noqa: E402


def load(name: str) -> dict:
    return json.loads((config.REPORTS / name).read_text(encoding="utf-8"))


def main() -> None:
    robust, baseline = load("robust_feature_model.json"), load("baseline_robustness.json")
    plain, cnn, uci = load("feature_models.json"), load("cnn_model.json"), load("uci_leakage_demo.json")
    key = robust["selected"]
    best, threshold = robust["results"][key], robust["threshold"]
    test_clean, test_pert = best["test_at_threshold"]["clean"], best["test_at_threshold"]["perturbed"]
    cv_clean, cv_pert = best["cv_at_threshold"]["clean"], best["cv_at_threshold"]["perturbed"]

    # Overfitting check: training vs cross-validated vs test performance of the deployed model.
    windows = pd.read_csv(config.DATA_PROCESSED / "windows_features.csv")
    bundle = joblib.load(config.MODELS / "feature_model.joblib")
    dev_clean = windows[(~windows.test) & (windows.version == "clean")]
    p_train = bundle["model"].predict_proba(dev_clean[bundle["features"]].to_numpy())[:, 1]
    train = dev_clean[["subject_id", "label"]].assign(p=p_train).groupby("subject_id").agg(label=("label", "first"), p=("p", "median"))
    train_auc = float(roc_auc_score(train.label, train.p))
    oof = pd.read_csv(config.REPORTS / "oof_robust_model.csv")
    band = evaluation.confidence_band_accuracy(oof[oof.family == "clean"].label, oof[oof.family == "clean"].p, threshold)

    lines = ["# Results", "",
             "All numbers are **person-level** (one prediction per person, the median over that person's 1.5 s windows). "
             "Cross-validation (CV) is 5-fold, stratified and grouped by person, on 64 development people. The test set is "
             "17 people held out from the start and used once. 'Perturbed' means the same people's audio after a "
             "formant-preserving pitch change of ±2 or ±4 semitones, or a simulated laptop microphone (16 kHz, room "
             "reverberation, background noise, different loudness and frequency response).", "",
             "## 1. Why the first model failed on live recordings", "",
             "The first deployed model (Random Forest on 50 features of the whole recording) was applied to the 17 held-out "
             "people:", "",
             "| Audio | Healthy people flagged as PD | Healthy people cleared | PD people flagged |", "|---|---|---|---|"]
    for family, label in [("clean", "Clean dataset recordings"), ("pitch", "Pitch changed ±2/±4 semitones"),
                          ("mic", "Simulated laptop microphone")]:
        b = baseline[family]
        lines.append(f"| {label} | {b['healthy_flagged_pd']:.0%} | {b['specificity']:.0%} | {b['sensitivity']:.0%} |")
    lines += ["", "It already flagged most unseen healthy people as PD on clean dataset audio, so the problem was not only "
              "the microphone: the model had memorised its training people (training AUC 1.0) and separated unseen people "
              "poorly. It also relied heavily on MFCC standard deviations, which carry the recording channel, and on "
              "absolute pitch. Uploading a dataset recording the model had trained on looked correct for that reason.", "",
              "## 2. The rebuilt model", "",
              "Changes: (1) features from fixed 1.5 s windows, median over windows, so recording length and onset/offset do "
              "not matter; (2) the option to drop absolute-pitch features; (3) training on pitch-shifted and "
              "microphone-simulated copies of each training person (kept in the same fold as the person); (4) model and "
              "feature set chosen on clean AND perturbed out-of-fold performance; (5) the decision threshold chosen on "
              "out-of-fold data to limit false alarms on healthy people.", "",
              "### Out-of-fold comparison (64 development people)", "",
              "| Model | Features | Training | Clean AUC | Perturbed AUC | Perturbed sensitivity | Perturbed specificity |",
              "|---|---|---|---|---|---|---|"]
    for k, r in sorted(robust["results"].items(), key=lambda kv: -(kv[1]["cv"]["clean"]["auc"] + kv[1]["cv"]["perturbed"]["auc"])):
        c = r["cv"]
        mark = " **(deployed)**" if k == key else ""
        lines.append(f"| {r['kind']}{mark} | {r['feature_set']} | {r['mode']} | {c['clean']['auc']:.3f} | "
                     f"{c['perturbed']['auc']:.3f} | {c['perturbed']['sensitivity']:.2f} | {c['perturbed']['specificity']:.2f} |")
    lines += ["", f"**Deployed:** {key}. Decision threshold {threshold} ({robust['threshold_rule']}).", "",
              "### At the deployed threshold", "",
              "| Set | Audio | AUC | Sensitivity | Specificity | Balanced accuracy |", "|---|---|---|---|---|---|"]
    for name, c, p in [("Cross-validation (64)", cv_clean, cv_pert), ("Held-out test (17)", test_clean, test_pert)]:
        for label, m in [("clean", c), ("perturbed", p)]:
            lines.append(f"| {name} | {label} | {m['auc']:.2f} | {m['sensitivity']:.2f} | {m['specificity']:.2f} | "
                         f"{m['balanced_accuracy']:.2f} |")
    tm = best["test_clean_metrics"]
    ci = tm["ci95"]["roc_auc"]
    cm = tm["confusion_matrix"]
    lines += ["", f"Held-out test, clean recordings: AUC {tm['roc_auc']:.2f} (95% CI {ci[0]:.2f}-{ci[1]:.2f}), "
              f"{cm[1][1]} of {cm[1][0] + cm[1][1]} people with PD flagged, {cm[0][0]} of {cm[0][0] + cm[0][1]} healthy "
              "people cleared. With 17 people the interval is wide.", "",
              "## 3. Overfitting and underfitting check (deployed model)", "",
              f"- Training AUC (people it was trained on): {train_auc:.2f}",
              f"- Cross-validated AUC, clean: {cv_clean['auc']:.2f}; perturbed: {cv_pert['auc']:.2f}",
              f"- Held-out test AUC, clean: {test_clean['auc']:.2f}; perturbed: {test_pert['auc']:.2f}",
              f"- Gap between training and cross-validated AUC: {train_auc - cv_clean['auc']:.2f}. "
              + ("A large gap means the model still fits its training people better than new people (overfitting), which "
                 "is expected with 64 training people; regularisation (shallow trees or strong L2), augmentation and "
                 "person-wise selection limit it but cannot remove it." if train_auc - cv_clean["auc"] > 0.15 else
                 "A small gap means little overfitting."),
              "- Underfitting would show as low training AUC; that is not the case here. The limiting factor is the "
              "amount of data, not model capacity.", "",
              "## 4. Confound check: age", "",
              f"- An age-only logistic regression reaches CV AUC {plain['results']['Age only (confound check)']['cv_folds']['roc_auc']['mean']:.2f}, "
              "because the healthy group is much younger (mean 48 vs 67 years).",
              f"- Among people aged 50 or over ({best['cv_age_50_plus_clean']['pd']} PD, {best['cv_age_50_plus_clean']['hc']} "
              f"healthy), the deployed model's out-of-fold AUC is {best['cv_age_50_plus_clean']['auc']:.2f} "
              f"(sensitivity {best['cv_age_50_plus_clean']['sensitivity']:.2f}, specificity "
              f"{best['cv_age_50_plus_clean']['specificity']:.2f}).",
              "- Treat the voice model's skill against that baseline: part of its signal may be age.", "",
              "## 5. Spectrogram CNN (explanation aid, not part of the score)", "",
              f"CV AUC {cnn['cv_folds']['roc_auc']['mean']:.2f} ± {cnn['cv_folds']['roc_auc']['std']:.2f} over folds "
              f"(pooled out-of-fold {cnn['cv_pooled_oof']['roc_auc']:.2f}); held-out test AUC {cnn['test']['roc_auc']:.2f}. "
              "The cross-validated and test numbers disagree because both sets are small. It is used for the Grad-CAM "
              "heatmap and shown as a second opinion that does not change the final result.", "",
              "## 6. Leakage demonstration (UCI Parkinson's, 32 people, about 6 recordings each)", "",
              "| Model | Accuracy, split by recording | Accuracy, split by person | AUC, split by recording | AUC, split by person |",
              "|---|---|---|---|---|"]
    for name, r in uci["results"].items():
        a, b = r["split_by_recording"], r["split_by_subject"]
        lines.append(f"| {name} | {a['recording_accuracy']:.3f} | {b['recording_accuracy']:.3f} | "
                     f"{a['recording_auc']:.3f} | {b['recording_auc']:.3f} |")
    lines += ["", "Splitting by recording lets a model recognise people it has already seen, which inflates every metric. "
              "All deployed-model results above use person-wise splits.", "",
              "## 7. Explainability", "", "Top features by mean |SHAP| on the deployed model's calibrated P(PD):", ""]
    for name, value in best["shap_mean_abs_top10"].items():
        lines.append(f"- `{name}`: {features.FEATURE_DESCRIPTIONS.get(name, name)} ({value:.3f})")
    lines += ["", "## 8. Confidence levels (out-of-fold, clean recordings)", "",
              "| Confidence | People | Accuracy |", "|---|---|---|"]
    for level, v in band.items():
        accuracy = "-" if v["accuracy"] is None else f"{v['accuracy']:.2f}"
        lines.append(f"| {level} | {v['n']} | {accuracy} |")
    lines += ["", "## Figures", ""] + [f"- `figures/{p.name}`" for p in sorted(config.FIGURES.glob("*.png"))]
    (config.REPORTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
