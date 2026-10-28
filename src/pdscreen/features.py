"""Clinically motivated voice features from a sustained vowel.

Praat (through Parselmouth) supplies pitch, jitter, shimmer and HNR with the
same algorithms used in the Little et al. UCI dataset. PPE, RPDE and DFA are
implemented here following Little et al. (2007, 2009); they are close
re-implementations, not bit-exact copies of the original MATLAB code.
"""

from __future__ import annotations

import numpy as np

from . import config

F0_FEATURES = ["f0_mean_hz", "f0_std_st", "f0_min_hz", "f0_max_hz", "f0_range_st", "voiced_fraction"]
JITTER_FEATURES = ["jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp"]
SHIMMER_FEATURES = ["shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5", "shimmer_apq11", "shimmer_dda"]
NOISE_FEATURES = ["hnr_db", "nhr"]
NONLINEAR_FEATURES = ["ppe", "rpde", "dfa"]
SPECTRAL_FEATURES = ["spectral_centroid_hz", "spectral_flatness"]
MFCC_FEATURES = [f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)]

FEATURE_NAMES = (F0_FEATURES + JITTER_FEATURES + SHIMMER_FEATURES + NOISE_FEATURES
                 + NONLINEAR_FEATURES + SPECTRAL_FEATURES + MFCC_FEATURES)

FEATURE_DESCRIPTIONS = {
    "f0_mean_hz": "Average pitch (Hz)",
    "f0_std_st": "Pitch variation (semitones); low values mean a monotone voice",
    "f0_min_hz": "Lowest pitch (Hz)",
    "f0_max_hz": "Highest pitch (Hz)",
    "f0_range_st": "Pitch range (semitones)",
    "voiced_fraction": "Share of the recording with a steady voice",
    "jitter_local": "Cycle-to-cycle pitch instability (%)",
    "jitter_abs": "Absolute jitter (seconds)",
    "jitter_rap": "Jitter, 3-period average",
    "jitter_ppq5": "Jitter, 5-period average",
    "jitter_ddp": "Jitter, difference of differences",
    "shimmer_local": "Cycle-to-cycle loudness instability",
    "shimmer_db": "Shimmer (dB)",
    "shimmer_apq3": "Shimmer, 3-period average",
    "shimmer_apq5": "Shimmer, 5-period average",
    "shimmer_apq11": "Shimmer, 11-period average",
    "shimmer_dda": "Shimmer, difference of differences",
    "hnr_db": "Harmonics-to-noise ratio (dB); low values mean a breathy or hoarse voice",
    "nhr": "Noise-to-harmonics ratio",
    "ppe": "Pitch period entropy: impaired pitch control",
    "rpde": "Recurrence period density entropy: voice periodicity",
    "dfa": "Detrended fluctuation analysis: turbulent noise in the voice",
    "spectral_centroid_hz": "Spectral brightness (Hz)",
    "spectral_flatness": "Spectral flatness (noisiness)",
}


def _f0_track(sound, floor: float, ceiling: float) -> tuple[object, np.ndarray]:
    pitch = sound.to_pitch_cc(time_step=0.01, pitch_floor=floor, pitch_ceiling=ceiling)
    return pitch, pitch.selected_array["frequency"]


def _adaptive_pitch(sound) -> tuple[object, np.ndarray, float, float]:
    """Two-pass pitch range (De Looze & Hirst) so one setting suits all voices."""
    pitch, f0 = _f0_track(sound, 60.0, 600.0)
    voiced = f0[f0 > 0]
    if voiced.size < 10:
        return pitch, f0, 60.0, 600.0
    floor = float(np.clip(0.75 * np.quantile(voiced, 0.25), 50.0, 300.0))
    ceiling = float(np.clip(1.5 * np.quantile(voiced, 0.75), floor + 50.0, 800.0))
    pitch, f0 = _f0_track(sound, floor, ceiling)
    return pitch, f0, floor, ceiling


def pitch_period_entropy(f0: np.ndarray, bins: int = 30) -> float:
    """PPE: entropy of the whitened semitone pitch sequence (Little et al., 2009)."""
    voiced = f0[f0 > 0]
    if voiced.size < 20:
        return float("nan")
    semitones = 12.0 * np.log2(voiced / 127.09)
    # Order-2 linear prediction removes smooth intonation; what is left is
    # the pitch-control noise PPE is meant to capture.
    X = np.column_stack([semitones[1:-1], semitones[:-2]])
    coefficients, *_ = np.linalg.lstsq(X, semitones[2:], rcond=None)
    residual = semitones[2:] - X @ coefficients
    hist, _ = np.histogram(residual, bins=bins)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum() / np.log(bins))


def rpde(x: np.ndarray, m: int = 4, tau: int = 7, epsilon: float = 0.12, t_max: int = 400,
         max_points: int = 8_000) -> float:
    """Recurrence period density entropy (Little et al., 2007), normalised to [0, 1].

    For each point of the delay embedding, the recurrence period is the time
    until the trajectory first returns inside the ``epsilon`` ball after
    having left it. A perfectly periodic voice gives one period (RPDE near 0);
    a noisy, aperiodic voice spreads the periods out (RPDE near 1).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size > max_points:
        start = (x.size - max_points) // 2
        x = x[start: start + max_points]
    x = x / (np.max(np.abs(x)) + 1e-12)
    n = x.size - (m - 1) * tau
    if n <= t_max + 10:
        return float("nan")
    embedded = np.column_stack([x[i * tau: i * tau + n] for i in range(m)])
    count = n - t_max
    base = embedded[:count]
    inside = np.empty((t_max, count), dtype=bool)
    for lag in range(1, t_max + 1):
        inside[lag - 1] = np.linalg.norm(embedded[lag: lag + count] - base, axis=1) < epsilon
    left = ~inside
    has_left = np.cumsum(left, axis=0) > 0
    returned = inside & np.vstack([np.zeros((1, count), dtype=bool), has_left[:-1]])
    found = returned.any(axis=0)
    periods = returned.argmax(axis=0)[found] + 1
    if periods.size == 0:
        return float("nan")
    hist = np.bincount(periods, minlength=t_max + 1)[1:]
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log(p)).sum() / np.log(t_max))


def dfa(x: np.ndarray, min_window: int = 16, max_window: int = 512, n_scales: int = 12) -> float:
    """Detrended fluctuation analysis scaling exponent (Little et al., 2007)."""
    x = np.asarray(x, dtype=np.float64)
    profile = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), n_scales).astype(int))
    fluctuations, used = [], []
    for scale in scales:
        segments = profile.size // scale
        if segments < 4:
            continue
        blocks = profile[: segments * scale].reshape(segments, scale)
        t = np.arange(scale)
        design = np.column_stack([t, np.ones(scale)])
        coeffs, *_ = np.linalg.lstsq(design, blocks.T, rcond=None)
        residual = blocks.T - design @ coeffs
        fluctuations.append(np.sqrt((residual ** 2).mean()))
        used.append(scale)
    if len(used) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(used), np.log(np.asarray(fluctuations) + 1e-12), 1)
    return float(slope)


def extract_features(y: np.ndarray, rate: int = config.SAMPLE_RATE) -> dict[str, float]:
    """Compute every feature in ``FEATURE_NAMES`` from a preprocessed signal."""
    import librosa
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=rate)
    pitch, f0, floor, ceiling = _adaptive_pitch(sound)
    voiced = f0[f0 > 0]
    out: dict[str, float] = {}

    if voiced.size >= 3:
        semitones = 12.0 * np.log2(voiced / 100.0)
        out.update(f0_mean_hz=float(voiced.mean()), f0_std_st=float(semitones.std()),
                   f0_min_hz=float(voiced.min()), f0_max_hz=float(voiced.max()),
                   f0_range_st=float(semitones.max() - semitones.min()))
    else:
        out.update({name: float("nan") for name in F0_FEATURES[:5]})
    out["voiced_fraction"] = float((f0 > 0).mean()) if f0.size else 0.0

    points = call([sound, pitch], "To PointProcess (cc)")
    jitter_args = (0.0, 0.0, 1.0 / ceiling, 1.0 / floor, 1.3)
    shimmer_args = jitter_args + (1.6,)
    for name, command in zip(JITTER_FEATURES, ["local", "local, absolute", "rap", "ppq5", "ddp"]):
        try:
            out[name] = float(call(points, f"Get jitter ({command})", *jitter_args))
        except Exception:
            out[name] = float("nan")
    for name, command in zip(SHIMMER_FEATURES, ["local", "local_dB", "apq3", "apq5", "apq11", "dda"]):
        try:
            out[name] = float(call([sound, points], f"Get shimmer ({command})", *shimmer_args))
        except Exception:
            out[name] = float("nan")

    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=floor)
    hnr_frames = harmonicity.values[harmonicity.values > -100]
    if hnr_frames.size:
        out["hnr_db"] = float(hnr_frames.mean())
        out["nhr"] = float(np.mean(10.0 ** (-hnr_frames / 10.0)))
    else:
        out["hnr_db"], out["nhr"] = float("nan"), float("nan")

    out["ppe"] = pitch_period_entropy(f0)
    out["rpde"] = rpde(y)
    out["dfa"] = dfa(y)

    spectrum = np.abs(librosa.stft(y, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)) ** 2
    out["spectral_centroid_hz"] = float(librosa.feature.spectral_centroid(S=spectrum, sr=rate).mean())
    out["spectral_flatness"] = float(librosa.feature.spectral_flatness(S=spectrum).mean())
    mel = librosa.feature.melspectrogram(S=spectrum, sr=rate, n_mels=40, fmax=config.BANDPASS_HZ[1])
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)
    for i in range(13):
        out[f"mfcc{i + 1}_mean"] = float(mfcc[i].mean())
        out[f"mfcc{i + 1}_std"] = float(mfcc[i].std())
    return {name: out[name] for name in FEATURE_NAMES}


def pitch_contour(y: np.ndarray, rate: int = config.SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Time (s) and F0 (Hz, NaN when unvoiced) for plotting."""
    import parselmouth

    sound = parselmouth.Sound(y.astype(np.float64), sampling_frequency=rate)
    pitch, f0, _, _ = _adaptive_pitch(sound)
    f0 = f0.astype(float)
    f0[f0 <= 0] = np.nan
    return pitch.xs(), f0
