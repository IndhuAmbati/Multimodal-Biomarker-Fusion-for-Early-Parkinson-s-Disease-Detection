"""Train and evaluate the log-Mel CNN with subject-wise cross-validation, then export to ONNX.

Same protocol as the feature models: subject-grouped stratified 5-fold CV on
the 64 development subjects (inside each training fold, 20% of subjects are
held back for early stopping), then one final model evaluated once on the 17
held-out test subjects. Combining it with the feature model happens in
``finalize.py``.

    python scripts/train_cnn.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import cnn, config, evaluation, explain  # noqa: E402


def split_for_early_stopping(subjects, labels, seed):
    return train_test_split(subjects, labels, test_size=0.2, stratify=labels, random_state=seed)


def plot_learning_curve(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(5.5, 3.6))
    ax1.plot(epochs, [h["train_loss"] for h in history], color="tab:blue", label="Training loss")
    ax1.plot(epochs, [h["val_loss"] for h in history], color="tab:blue", linestyle="--", label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (solid: train, dashed: validation)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_auc"] for h in history], color="tab:orange", label="Validation AUC (subjects)")
    ax2.set_ylabel("Validation ROC-AUC", color="tab:orange")
    ax2.set_ylim(0, 1)
    fig.suptitle("Final CNN: training loss and early-stopping signal", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gradcam(model, front, signals, examples: list[tuple[str, str]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 2, figsize=(8, 2.6 * len(examples)), squeeze=False)
    for row, (subject, title) in enumerate(examples):
        wave = cnn.windows(signals[subject])[0]
        mel = front(torch.tensor(wave[None]))[0].numpy()
        cam = explain.grad_cam(model, mel)
        axes[row, 0].imshow(mel, origin="lower", aspect="auto", cmap="magma")
        axes[row, 0].set_title(f"{title}: log-Mel (1 s)", fontsize=9)
        axes[row, 1].imshow(mel, origin="lower", aspect="auto", cmap="gray")
        axes[row, 1].imshow(cam, origin="lower", aspect="auto", cmap="jet", alpha=0.45)
        axes[row, 1].set_title("Grad-CAM: regions that raised P(PD)", fontsize=9)
        for ax in axes[row]:
            ax.set_xlabel("Time frames (10 ms)")
            ax.set_ylabel("Mel band")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(folds: int = 5, repeats: int = 1) -> None:
    torch.set_num_threads(4)
    table = pd.read_csv(config.DATA_PROCESSED / "figshare_features.csv")
    signals = dict(np.load(config.DATA_PROCESSED / "figshare_signals.npz"))
    dev, test = table[~table.test].reset_index(drop=True), table[table.test].reset_index(drop=True)
    subjects, labels = dev.subject_id.to_numpy(), dev.label.to_numpy()

    oof = np.zeros((repeats, len(dev)))
    fold_index, fold_training = [], []
    start = time.perf_counter()
    for r in range(repeats):
        outer = StratifiedGroupKFold(folds, shuffle=True, random_state=config.SEED + r)
        for k, (train, held) in enumerate(outer.split(subjects, labels, subjects)):
            tr_s, va_s, tr_y, va_y = split_for_early_stopping(subjects[train], labels[train], config.SEED + r)
            model, info = cnn.train_model(signals, tr_s, tr_y, va_s, va_y, seed=config.SEED + r)
            front = cnn.LogMel()
            waves, _, owners = cnn.build_windows(signals, subjects[held], labels[held], augment=False)
            oof[r, held] = cnn.subject_probabilities(model, front, waves, owners, len(held))
            fold_index.append((r, held))
            fold_training.append({k2: v for k2, v in info.items() if k2 != "history"})
            print(f"repeat {r + 1} fold {k + 1}: AUC {evaluation.classification_metrics(labels[held], oof[r, held])['roc_auc']:.3f} "
                  f"(best epoch {info['best_epoch']}/{info['epochs_run']}, val loss {info['best_val_loss']:.3f})", flush=True)
    cv_seconds = time.perf_counter() - start

    # Calibrate the CNN's raw probabilities. For honest CV numbers each fold is
    # calibrated with a Platt scaler fitted on the other folds' predictions
    # (cross-fitting); the deployed calibrator is fitted on all out-of-fold
    # predictions and is then applied, unchanged, to the test set.
    oof_cal = np.zeros_like(oof)
    for r, held in fold_index:
        others = np.setdiff1d(np.arange(len(dev)), held)
        oof_cal[r, held] = cnn.apply_platt(oof[r, held], cnn.fit_platt(oof[r, others], labels[others]))
    fold_metrics = [evaluation.classification_metrics(labels[held], oof_cal[r, held]) for r, held in fold_index]
    p_oof_raw, p_oof = oof.mean(axis=0), oof_cal.mean(axis=0)
    calibration = cnn.fit_platt(p_oof_raw, labels)

    # Final model on all development subjects (20% for early stopping), one test evaluation.
    tr_s, va_s, tr_y, va_y = split_for_early_stopping(subjects, labels, config.SEED)
    final, info = cnn.train_model(signals, tr_s, tr_y, va_s, va_y)
    front = cnn.LogMel()
    waves, _, owners = cnn.build_windows(signals, test.subject_id.to_numpy(), test.label.to_numpy(), augment=False)
    p_test_raw = cnn.subject_probabilities(final, front, waves, owners, len(test))
    p_test = cnn.apply_platt(p_test_raw, calibration)
    test_metrics = evaluation.classification_metrics(test.label.to_numpy(), p_test)
    test_metrics["ci95"] = evaluation.bootstrap_ci(test.label.to_numpy(), p_test)
    print("Held-out test:", {k: round(v, 3) for k, v in test_metrics.items() if isinstance(v, float)})

    # Export: PyTorch weights + ONNX body, and check that ONNX matches PyTorch.
    config.MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(final.state_dict(), config.MODELS / "mel_cnn.pt")
    cnn.export_onnx(final, config.MODELS / "mel_cnn.onnx")
    (config.MODELS / "cnn_calibration.json").write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    import onnxruntime as ort
    session = ort.InferenceSession(str(config.MODELS / "mel_cnn.onnx"), providers=["CPUExecutionProvider"])
    mel = front(torch.tensor(waves[:8]))[:, None]
    with torch.no_grad():
        torch_out = final(mel).numpy()
    onnx_out = session.run(None, {"log_mel": mel.numpy()})[0]
    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"ONNX vs PyTorch max logit difference: {max_diff:.2e}")

    # Figures.
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_learning_curve(info["history"], config.FIGURES / "cnn_learning_curve.png")
    pd_example = test[test.label == 1].subject_id.iloc[0]
    hc_example = test[test.label == 0].subject_id.iloc[0]
    plot_gradcam(final, front, signals, [(pd_example, "PD (test)"), (hc_example, "Healthy (test)")],
                 config.FIGURES / "gradcam_examples.png")
    evaluation.plot_confusion(test_metrics["confusion_matrix"], "Mel-CNN: held-out test (17)",
                              config.FIGURES / "confusion_cnn_test.png")

    older = dev.age.to_numpy() >= 50
    report = {
        "architecture": "4-block CNN (16-32-64-64 channels) on 64-band log-Mel, 1 s windows, mean over windows",
        "parameters": int(sum(p.numel() for p in final.parameters())),
        "training": {"loss": "BCE with pos_weight", "optimizer": "AdamW lr 1e-3, wd 1e-3",
                     "schedule": "cosine annealing", "max_epochs": 100, "min_epochs": 10,
                     "early_stopping": "validation window loss, patience 15",
                     "augmentation": "pitch shift +/-0.5, +/-1 semitone; additive noise 15-40 dB SNR; SpecAugment",
                     "calibration": "Platt scaling on out-of-fold predictions (cross-fitted for CV metrics)"},
        "calibration": calibration,
        "cv_folds": evaluation.summarise_folds(fold_metrics),
        "cv_pooled_oof": evaluation.classification_metrics(labels, p_oof),
        "cv_pooled_oof_uncalibrated_auc": evaluation.classification_metrics(labels, p_oof_raw)["roc_auc"],
        "cv_oof_age_50_plus": evaluation.classification_metrics(labels[older], p_oof[older]),
        "fold_training": fold_training,
        "cv_seconds": round(cv_seconds, 1),
        "final_training": {k: v for k, v in info.items() if k != "history"},
        "test": test_metrics,
        "confidence_band_accuracy_oof": evaluation.confidence_band_accuracy(labels, p_oof),
        "onnx_max_abs_logit_diff": max_diff,
    }
    pd.DataFrame({"subject_id": subjects, "label": labels, "age": dev.age, "p_cnn": p_oof, "p_cnn_raw": p_oof_raw}) \
        .to_csv(config.REPORTS / "oof_cnn.csv", index=False)
    pd.DataFrame({"subject_id": test.subject_id, "label": test.label, "p_cnn": p_test, "p_cnn_raw": p_test_raw}) \
        .to_csv(config.REPORTS / "test_predictions_cnn.csv", index=False)
    (config.REPORTS / "cnn_model.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("CV AUC", round(report["cv_folds"]["roc_auc"]["mean"], 3), "+/-", round(report["cv_folds"]["roc_auc"]["std"], 3),
          "| pooled OOF AUC", round(report["cv_pooled_oof"]["roc_auc"], 3), "| test AUC", round(test_metrics["roc_auc"], 3))


if __name__ == "__main__":
    main()
