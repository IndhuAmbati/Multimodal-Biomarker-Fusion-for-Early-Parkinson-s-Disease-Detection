"""SHAP explanations for the feature model and Grad-CAM for the spectrogram CNN.

pandas is only needed by the SHAP functions and is imported inside them, so Grad-CAM (and importing this module) works
on machines where pandas is blocked.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def _probability_fn(model):
    return lambda X: model.predict_proba(np.asarray(X, dtype=float))[:, 1]


def kernel_explainer(model, background: pd.DataFrame, clusters: int = 10):
    """Model-agnostic SHAP on the calibrated P(PD), so explanations match the shown score."""
    import shap  # noqa: F401  (needs pandas)

    filled = background.fillna(background.median())
    summary = shap.kmeans(filled.to_numpy(dtype=float), min(clusters, len(filled)))
    return shap.KernelExplainer(_probability_fn(model), summary)


def global_shap(model, background: pd.DataFrame, columns: list[str], path: Path, nsamples: int = 200) -> dict:
    """Beeswarm plot over the development subjects; returns the top-10 mean |SHAP|."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import shap

    explainer = kernel_explainer(model, background)
    X = background.fillna(background.median())
    values = explainer.shap_values(X.to_numpy(dtype=float), nsamples=nsamples, silent=True)
    shap.summary_plot(values, X, feature_names=columns, show=False, max_display=12, plot_size=(7, 5))
    plt.title("SHAP: how each feature moves P(PD)", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close("all")
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=columns).sort_values(ascending=False)
    return {k: round(float(v), 4) for k, v in mean_abs.head(10).items()}


def local_shap(explainer, row: pd.DataFrame, nsamples: int = 150) -> pd.Series:
    """SHAP values for one recording, sorted by absolute contribution."""
    import pandas as pd

    values = explainer.shap_values(row.to_numpy(dtype=float), nsamples=nsamples, silent=True)
    series = pd.Series(np.asarray(values).reshape(-1), index=row.columns)
    return series.reindex(series.abs().sort_values(ascending=False).index)


def grad_cam(model, spectrogram: np.ndarray) -> np.ndarray:
    """Grad-CAM heatmap (same shape as the input log-Mel window) for the PD logit.

    ``model`` is the PyTorch CNN; its last convolutional block is exposed as
    ``model.last_conv``.
    """
    import torch

    activations, gradients = {}, {}
    layer = model.last_conv
    forward = layer.register_forward_hook(lambda _, __, out: activations.__setitem__("a", out))
    backward = layer.register_full_backward_hook(lambda _, __, grad: gradients.__setitem__("g", grad[0]))
    try:
        model.eval()
        x = torch.tensor(spectrogram, dtype=torch.float32)[None, None]
        x.requires_grad_(True)
        model.zero_grad()
        model(x).sum().backward()
        weights = gradients["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations["a"]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=spectrogram.shape, mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
        return cam / (cam.max() + 1e-9)
    finally:
        forward.remove()
        backward.remove()
