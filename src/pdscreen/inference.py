"""Real-time screening: audio bytes -> preprocessing -> features + CNN -> calibrated P(PD).

``Screener`` loads the deployed models once (feature model via joblib, CNN
via ONNX Runtime) and is shared by the Streamlit app and the FastAPI service.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, compat, config, features
from .evaluation import confidence_level


@dataclass
class ScreeningResult:
    label: str
    probability: float
    confidence: str
    components: dict[str, float]
    features: dict[str, float]
    out_of_range: list[str]
    quality: dict
    latency_ms: dict[str, float]
    model_info: dict
    disclaimer: str = config.DISCLAIMER
    notes: list[str] = field(default_factory=list)
    secondary: dict[str, float] = field(default_factory=dict)
    window_probs: list[float] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "status": "completed",
            "label": self.label,
            "probability_pd": round(self.probability, 4),
            "probability_healthy": round(1 - self.probability, 4),
            "confidence": self.confidence,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "secondary_not_in_score": {k: round(v, 4) for k, v in self.secondary.items()},
            "window_probabilities": [round(float(p), 4) for p in self.window_probs],
            "features": {k: (None if not np.isfinite(v) else round(float(v), 6)) for k, v in self.features.items()},
            "out_of_training_range": self.out_of_range,
            "quality": self.quality,
            "latency_ms": self.latency_ms,
            "model": self.model_info,
            "notes": self.notes,
            "disclaimer": self.disclaimer,
        }


class Screener:
    def __init__(self, model_dir: Path = config.MODELS):
        compat.pandas_available()  # settles the pandas question once, before scikit-learn probes for it
        import joblib
        import onnxruntime as ort

        from .cnn import LogMel

        self.model_dir = Path(model_dir)
        deployment_path = self.model_dir / "deployment.json"
        if not deployment_path.exists():
            raise FileNotFoundError("models/deployment.json is missing. Run the training scripts first.")
        self.deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        self.components = self.deployment["components"]
        self.threshold = float(self.deployment.get("threshold", 0.5))
        self.feature_bundle = joblib.load(self.model_dir / "feature_model.joblib") if "feature_model" in self.components else None
        # The CNN is loaded whenever it exists: it scores only if deployed, but
        # always provides the spectrogram Grad-CAM and per-second view.
        cnn_path = self.model_dir / "mel_cnn.onnx"
        self.session = (ort.InferenceSession(str(cnn_path), providers=["CPUExecutionProvider"])
                        if cnn_path.exists() else None)
        self.front = LogMel()
        calibration_path = self.model_dir / "cnn_calibration.json"
        self.cnn_calibration = json.loads(calibration_path.read_text(encoding="utf-8")) \
            if self.session is not None and calibration_path.exists() else None
        self._explainer = None
        self._pool = ThreadPoolExecutor(max_workers=4)
        if self.feature_bundle is not None:  # single-threaded forests: thread-pool start-up costs more than it saves
            for step in getattr(self.feature_bundle["model"], "calibrated_classifiers_", []):
                if hasattr(step.estimator, "named_steps") and "clf" in step.estimator.named_steps:
                    clf = step.estimator.named_steps["clf"]
                    if hasattr(clf, "n_jobs"):
                        clf.n_jobs = 1

    def warm_up(self) -> None:
        """Run once at start-up so the first real request does not pay library JIT costs."""
        t = np.arange(3 * config.SAMPLE_RATE) / config.SAMPLE_RATE
        tone = (0.3 * np.sin(2 * np.pi * 150 * t) + 0.005 * np.random.default_rng(0).standard_normal(t.size))
        self.predict_signal(tone.astype(np.float32), config.SAMPLE_RATE)

    # ------------------------------------------------------------ components
    def _feature_probabilities(self, window_values: list[dict[str, float]]) -> list[float]:
        """One batched call for all windows (separate calls cost about 80 ms each)."""
        bundle = self.feature_bundle
        matrix = np.array([[v[c] for c in bundle["features"]] for v in window_values], dtype=float)
        return [float(p) for p in bundle["model"].predict_proba(matrix)[:, 1]]

    def _cnn_probability(self, y: np.ndarray) -> float:
        import torch

        from .cnn import apply_platt, windows

        with torch.no_grad():
            mel = self.front(torch.tensor(windows(y)))[:, None].numpy()
        logits = self.session.run(None, {"log_mel": mel})[0]
        return float(apply_platt((1 / (1 + np.exp(-logits))).mean(), self.cnn_calibration))

    def log_mel(self, y: np.ndarray) -> np.ndarray:
        """Full-length log-Mel image of the cleaned recording, for display."""
        import torch

        with torch.no_grad():
            return self.front(torch.tensor(y[None]))[0].numpy()

    # ------------------------------------------------------------ public API
    def predict_bytes(self, contents: bytes) -> tuple[ScreeningResult, np.ndarray]:
        raw, rate = audio.load_audio(contents)
        return self.predict_signal(raw, rate)

    def predict_signal(self, raw: np.ndarray, rate: int) -> tuple[ScreeningResult, np.ndarray]:
        timings = {}
        t0 = time.perf_counter()
        y, quality = audio.preprocess(raw, rate)
        timings["preprocess"] = (time.perf_counter() - t0) * 1000

        # Score fixed-length windows and take the median, so recording length, onset/offset transients and
        # pitch drift cannot dominate; the representative window supplies the displayed measurements.
        t1 = time.perf_counter()
        window_values = list(self._pool.map(features.extract_features,
                                            audio.analysis_windows(y, max_windows=config.INFERENCE_MAX_WINDOWS)))
        timings["features"] = (time.perf_counter() - t1) * 1000

        t2 = time.perf_counter()
        components, secondary, window_probs = {}, {}, []
        if self.feature_bundle is not None:
            window_probs = self._feature_probabilities(window_values)
            components["feature_model"] = float(np.median(window_probs))
        representative = int(np.argmin(np.abs(np.array(window_probs) - np.median(window_probs)))) if window_probs else 0
        values = window_values[representative]
        if self.session is not None:
            (components if "cnn" in self.components else secondary)["cnn"] = self._cnn_probability(y)
        probability = float(np.mean(list(components.values())))
        timings["models"] = (time.perf_counter() - t2) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        out_of_range = []
        if self.feature_bundle is not None:
            for name, (low, high) in self.feature_bundle["training_ranges"].items():
                if np.isfinite(values[name]) and not (low <= values[name] <= high):
                    out_of_range.append(name)

        confidence = confidence_level(probability, self.threshold)
        notes = []
        if len(window_probs) > 1 and np.ptp(window_probs) > 0.35:
            notes.append("The voice looked different from one part of the recording to another (the 1.5 s windows "
                         "disagree). Hold one steady 'aaah' without changing pitch or loudness.")
            if confidence == "High":
                confidence = "Moderate"
        if quality.warnings or len(out_of_range) > 3:
            if confidence == "High":
                confidence = "Moderate"
            notes.append("Confidence was lowered because of recording quality or unusual voice measurements.")
        if confidence == "Low":
            notes.append("The score is close to the decision threshold; record again in a quiet room.")
        result = ScreeningResult(
            label="PD likely" if probability >= self.threshold else "Healthy",
            probability=probability, confidence=confidence, components=components, features=values,
            out_of_range=out_of_range, quality=quality.as_dict(),
            latency_ms={k: round(v, 1) for k, v in timings.items()},
            model_info={k: self.deployment[k] for k in ("name", "evaluation", "threshold") if k in self.deployment},
            notes=notes, secondary=secondary, window_probs=window_probs,
        )
        return result, y

    def explain_features(self, values: dict[str, float], top: int = 8) -> tuple[list[tuple[str, float]], str]:
        """Which voice measurements pushed this recording's P(PD) up or down.

        Returns ([(feature, contribution), ...] sorted by size, method). Uses SHAP when it can be loaded; otherwise
        one-at-a-time occlusion: each measurement is replaced by the training median and the drop in P(PD) is the
        measurement's contribution. Positive means the measurement raised P(PD).
        """
        bundle = self.feature_bundle
        names = bundle["features"]
        background = np.array([bundle["background"][c] for c in names], dtype=float)
        x = np.array([values[c] for c in names], dtype=float)
        x = np.where(np.isfinite(x), x, background)
        if compat.shap_available():
            import pandas as pd

            from .explain import kernel_explainer, local_shap

            if self._explainer is None:
                self._explainer = kernel_explainer(bundle["model"], pd.DataFrame(bundle["shap_background"], columns=names))
            series = local_shap(self._explainer, pd.DataFrame([x], columns=names)).head(top)
            return list(zip(series.index.tolist(), series.to_numpy(dtype=float).tolist())), "SHAP"
        rows = np.tile(x, (len(names) + 1, 1))
        rows[np.arange(1, len(names) + 1), np.arange(len(names))] = background
        p = bundle["model"].predict_proba(rows)[:, 1]
        contributions = p[0] - p[1:]
        order = np.argsort(-np.abs(contributions))[:top]
        return [(names[i], float(contributions[i])) for i in order], "occlusion"

    def grad_cam(self, y: np.ndarray) -> dict:
        """Grad-CAM across the whole recording, using the PyTorch twin of the ONNX model.

        Each 1-second window gets its own Grad-CAM map, scaled by that window's
        P(PD) so windows the CNN found healthy-looking stay dark. Overlapping
        windows are averaged onto the full-length log-Mel spectrogram. Also
        returns the CNN's P(PD) for every window, to show where in time the
        voice looked most PD-like.
        """
        import torch

        from .cnn import HOP, MelCNN, apply_platt, windows
        from .explain import grad_cam

        if getattr(self, "_torch_cnn", None) is None:
            model = MelCNN()
            model.load_state_dict(torch.load(self.model_dir / "mel_cnn.pt", map_location="cpu"))
            model.eval()
            self._torch_cnn = model
        model = self._torch_cnn
        waves = windows(y)
        with torch.no_grad():
            mels = self.front(torch.tensor(waves)).numpy()
            probs = apply_platt(torch.sigmoid(model(torch.tensor(mels)[:, None])).numpy(), self.cnn_calibration)
        full = self.log_mel(y)
        count = np.zeros(full.shape[1])
        heat2d = np.zeros_like(full)
        step = HOP // config.HOP_LENGTH
        for i, (mel, p) in enumerate(zip(mels, probs)):
            cam = grad_cam(model, mel) * float(p)
            start = i * step
            end = min(full.shape[1], start + cam.shape[1])
            heat2d[:, start:end] += cam[:, : end - start]
            count[start:end] += 1
        heat2d /= np.maximum(count, 1)[None, :]
        heat = heat2d.mean(axis=0)
        centres = (np.arange(len(waves)) * HOP + len(waves[0]) / 2) / config.SAMPLE_RATE
        return {"log_mel": full, "cam": heat2d, "cam_time_profile": heat, "window_times": centres,
                "window_probabilities": probs}
