"""Audio loading and the single preprocessing chain used everywhere.

Training recordings and live microphone recordings both pass through
``preprocess``: mono -> resample to 8 kHz -> band-pass -> spectral-gating
denoise -> silence trim -> peak normalisation. Keeping one chain avoids the
train/inference mismatch that makes live predictions drift.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sps

from . import config


class AudioError(ValueError):
    """Raised when a recording cannot be analysed."""


@dataclass
class Quality:
    duration_seconds: float
    voiced_seconds: float
    original_sample_rate: int
    snr_db: float
    clipped_fraction: float
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "voiced_seconds": round(self.voiced_seconds, 2),
            "original_sample_rate": self.original_sample_rate,
            "snr_db": round(self.snr_db, 1) if np.isfinite(self.snr_db) else None,
            "clipped_fraction": round(self.clipped_fraction, 4),
            "warnings": list(self.warnings),
        }


def load_audio(source: str | Path | bytes) -> tuple[np.ndarray, int]:
    """Read a WAV/FLAC/OGG file or bytes into a float32 mono signal."""
    try:
        data, rate = sf.read(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source,
                             dtype="float32", always_2d=True)
    except Exception as error:  # soundfile raises several error types
        raise AudioError("This file could not be read as audio. Please use a WAV recording.") from error
    if data.size == 0:
        raise AudioError("The recording is empty.")
    mono = data.mean(axis=1)
    if not np.all(np.isfinite(mono)):
        raise AudioError("The recording contains invalid samples.")
    return mono, int(rate)


def resample(x: np.ndarray, rate: int, target: int = config.SAMPLE_RATE) -> np.ndarray:
    if rate == target:
        return x.astype(np.float32)
    divisor = np.gcd(rate, target)
    return sps.resample_poly(x, target // divisor, rate // divisor).astype(np.float32)


def bandpass(x: np.ndarray, rate: int = config.SAMPLE_RATE) -> np.ndarray:
    low, high = config.BANDPASS_HZ
    high = min(high, 0.45 * rate)
    sos = sps.butter(4, [low, high], btype="bandpass", fs=rate, output="sos")
    return sps.sosfiltfilt(sos, x).astype(np.float32)


def spectral_gate(x: np.ndarray, rate: int = config.SAMPLE_RATE, strength: float = 1.0) -> np.ndarray:
    """Stationary spectral-gating noise reduction.

    The noise profile is the quietest 10% of frames, so it works on a
    sustained vowel that has no leading silence. Bins under the profile plus
    a 1.5-sigma margin are attenuated rather than zeroed, which keeps the
    harmonic structure that jitter and shimmer depend on.
    """
    nperseg = 256
    _, _, spec = sps.stft(x, fs=rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    magnitude = np.abs(spec)
    frame_energy = magnitude.mean(axis=0)
    quiet = magnitude[:, frame_energy <= np.quantile(frame_energy, 0.10)]
    if quiet.shape[1] == 0:
        return x
    noise_mean = quiet.mean(axis=1, keepdims=True)
    noise_std = quiet.std(axis=1, keepdims=True)
    threshold = noise_mean + 1.5 * noise_std
    mask = np.clip((magnitude - threshold) / (magnitude + 1e-9), 0.0, 1.0)
    mask = sps.convolve2d(mask, np.ones((3, 3)) / 9.0, mode="same")
    gain = 1.0 - strength * (1.0 - np.maximum(mask, 0.1))
    _, cleaned = sps.istft(spec * gain, fs=rate, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    return cleaned[: len(x)].astype(np.float32)


def active_span(x: np.ndarray, rate: int = config.SAMPLE_RATE, top_db: float = 30.0) -> tuple[int, int]:
    """Sample range from the first to the last frame within ``top_db`` of the peak."""
    frame, hop = int(0.025 * rate), int(0.010 * rate)
    if len(x) < frame:
        return 0, len(x)
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop]
    rms_db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-10)
    active = np.flatnonzero(rms_db > rms_db.max() - top_db)
    if active.size == 0:
        return 0, 0
    return int(active[0] * hop), int(min(len(x), active[-1] * hop + frame))


def estimate_snr_db(x: np.ndarray, start: int, end: int, rate: int = config.SAMPLE_RATE) -> float:
    """Voice-to-background ratio, or NaN when there is too little background to measure.

    Training recordings are already trimmed to the vowel, so they have no
    pause to measure noise from; a live recording normally does.
    """
    background = np.concatenate([x[:start], x[end:]])
    if len(background) < int(0.2 * rate) or end <= start:
        return float("nan")
    voice = x[start:end]
    return float(10 * np.log10((np.mean(voice ** 2) + 1e-12) / (np.mean(background ** 2) + 1e-12)))


def preprocess(x: np.ndarray, rate: int, denoise: bool = True) -> tuple[np.ndarray, Quality]:
    """Return the cleaned 8 kHz signal and a quality report."""
    duration = len(x) / rate
    if duration > config.MAX_INPUT_SECONDS:
        x = x[: int(config.MAX_INPUT_SECONDS * rate)]
    clipped = float(np.mean(np.abs(x) >= 0.999))
    y = resample(x - x.mean(), rate)
    y = bandpass(y)
    start, end = active_span(y)
    snr = estimate_snr_db(y, start, end)
    if denoise:
        y = spectral_gate(y)
    y = y[start:end]
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak < 1e-4:
        raise AudioError("No voice was detected. Record again closer to the microphone.")
    y = (0.95 * y / peak).astype(np.float32)
    voiced = len(y) / config.SAMPLE_RATE

    warnings = []
    if voiced < config.MIN_VOICED_SECONDS:
        raise AudioError(f"Only {voiced:.1f} s of voice was found. Hold the 'aaah' for at least 3 seconds.")
    if voiced < 3.0:
        warnings.append("Short recording: hold the 'aaah' for 5-10 seconds for a more reliable result.")
    if np.isfinite(snr) and snr < 15.0:
        warnings.append(f"Background noise is high (SNR {snr:.0f} dB). Record in a quieter room.")
    if clipped > 0.01:
        warnings.append("The recording is clipped (too loud). Move slightly away from the microphone.")
    return y, Quality(duration, voiced, rate, snr, clipped, warnings)


def analysis_windows(y: np.ndarray, seconds: float = config.ANALYSIS_WINDOW_SECONDS,
                     hop: float = config.ANALYSIS_HOP_SECONDS, max_windows: int | None = None) -> list[np.ndarray]:
    """Overlapping fixed-length windows, each scaled to the same RMS level.

    Equal length and level make window features comparable between a 3 s
    telephone clip and a 10 s live recording. ``max_windows`` keeps latency
    bounded by sampling evenly across the recording.
    """
    size, step = int(seconds * config.SAMPLE_RATE), int(hop * config.SAMPLE_RATE)
    if len(y) <= size:
        segments = [y]
    else:
        starts = list(range(0, len(y) - size + 1, step))
        if starts[-1] + size < len(y):
            starts.append(len(y) - size)
        if max_windows and len(starts) > max_windows:
            picks = sorted(set(np.linspace(0, len(starts) - 1, max_windows).round().astype(int).tolist()))
            starts = [starts[i] for i in picks]
        segments = [y[s: s + size] for s in starts]
    return [(s * (config.WINDOW_TARGET_RMS / (np.sqrt(np.mean(s ** 2)) + 1e-9))).astype(np.float32) for s in segments]
