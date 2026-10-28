"""Subject-level metrics, bootstrap intervals, calibration and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            error += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(error)


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "sensitivity": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": threshold,
    }


def bootstrap_ci(y, p, threshold: float = 0.5, n_boot: int = 2000, seed: int = 0,
                 keys=("accuracy", "sensitivity", "specificity", "f1", "roc_auc")) -> dict:
    """Percentile 95% intervals, resampling subjects with replacement."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = classification_metrics(y[idx], p[idx], threshold)
        for k in keys:
            samples[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}


def summarise_folds(fold_metrics: list[dict], keys=("accuracy", "precision", "sensitivity", "specificity",
                                                    "f1", "roc_auc", "brier")) -> dict:
    return {k: {"mean": float(np.nanmean([m[k] for m in fold_metrics])),
                "std": float(np.nanstd([m[k] for m in fold_metrics]))} for k in keys}


def confidence_level(p: float, threshold: float = 0.5) -> str:
    """Plain-language confidence: how far the calibrated probability is from the decision threshold."""
    distance = abs(p - threshold)
    if distance >= 0.25:
        return "High"
    if distance >= 0.10:
        return "Moderate"
    return "Low"


def confidence_band_accuracy(y, p, threshold: float = 0.5) -> dict:
    """How often the model is right inside each confidence band (out-of-fold)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    levels = np.array([confidence_level(v, threshold) for v in p])
    result = {}
    for level in ["High", "Moderate", "Low"]:
        mask = levels == level
        result[level] = {"n": int(mask.sum()),
                         "accuracy": float(((p[mask] >= threshold) == y[mask]).mean()) if mask.any() else None}
    return result


# ----------------------------------------------------------------- plots

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(cm, title: str, path: Path) -> None:
    plt = _plt()
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=14,
                color="white" if value > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Healthy", "PD"])
    ax.set_yticks([0, 1], ["Healthy", "PD"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str, bins: int = 5) -> None:
    from sklearn.calibration import calibration_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Perfect calibration")
    for name, (y, p) in curves.items():
        frac, mean = calibration_curve(y, p, n_bins=bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", label=f"{name} (Brier {brier_score_loss(y, p):.3f})")
    ax.set_xlabel("Predicted probability of PD")
    ax.set_ylabel("Observed share with PD")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
