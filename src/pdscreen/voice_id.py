"""Estimate whether an unknown voice sounds male or female.

The model reads the same cleaned, windowed audio and features as the screening model (pitch plus the 13 MFCC
means), so it works from a microphone or an uploaded file. It describes how a voice *sounds*, not who someone
is: children, people with voice disorders and unusually high or low voices can be misjudged.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np

from . import audio, config, features

MFCC_MEANS = [f"mfcc{i}_mean" for i in range(1, 14)]
INPUT_FEATURES = ["log2_f0"] + MFCC_MEANS
UNCERTAIN_BAND = (0.35, 0.65)
TYPICAL_PITCH_HZ = {"male": (85, 155), "female": (165, 255)}  # adult sustained vowels (Titze, 1994)


class AveragedModel:
    """Mean of several probability models, each reading the same feature matrix.

    One model sees pitch only and one sees pitch plus spectral shape (MFCCs). Pitch is a steady cue across
    microphones; MFCCs add accuracy on familiar audio but can mislead on unfamiliar audio. Averaging makes a
    confident answer require both to agree.
    """

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1.0 - p, p])


def feature_matrix(window_values: list[dict[str, float]]) -> np.ndarray:
    """Rows of [log2 F0, 13 MFCC means]; windows with no measurable pitch carry NaN (imputed by the model)."""
    rows = []
    for v in window_values:
        f0 = v["f0_mean_hz"]
        rows.append([np.log2(f0) if np.isfinite(f0) and f0 > 40 else np.nan, *[v[name] for name in MFCC_MEANS]])
    return np.array(rows, dtype=float)


class VoiceIdentifier:
    def __init__(self, model_dir: Path = config.MODELS):
        self.bundle = joblib.load(Path(model_dir) / "voice_id.joblib")
        self.band = tuple(self.bundle.get("uncertain_band", UNCERTAIN_BAND))
        self._pool = ThreadPoolExecutor(max_workers=4)

    def predict_bytes(self, contents: bytes) -> dict:
        raw, rate = audio.load_audio(contents)
        return self.predict_signal(raw, rate)

    def predict_signal(self, raw: np.ndarray, rate: int) -> dict:
        start = time.perf_counter()
        y, quality = audio.preprocess(raw, rate)
        windows = audio.analysis_windows(y, max_windows=config.INFERENCE_MAX_WINDOWS)
        values = list(self._pool.map(features.extract_features, windows))
        probabilities = self.bundle["model"].predict_proba(feature_matrix(values))[:, 1]
        p_female = float(np.median(probabilities))
        f0 = np.array([v["f0_mean_hz"] for v in values], dtype=float)
        f0 = f0[np.isfinite(f0)]
        low, high = self.band
        if p_female >= high:
            label = "Likely female"
        elif p_female <= low:
            label = "Likely male"
        else:
            label = "Uncertain"
        return {
            "label": label,
            "probability_female": p_female,
            "probability_male": 1.0 - p_female,
            "pitch_hz": float(np.median(f0)) if f0.size else None,
            "windows": len(values),
            "quality": quality.as_dict(),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
