"""Pitch and microphone-channel perturbations, for robustness training and testing.

The training data are telephone recordings. A live laptop or phone microphone
adds a different frequency response, room reverberation, background noise,
silence before and after the vowel, a different loudness, and the speaker's
own choice of pitch. These functions recreate those differences from a
clean recording so models can be trained on them and measured against them.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy import signal as sps

from . import audio

VERSIONS = ["clean", "pitch-4", "pitch-2", "pitch+2", "pitch+4", "mic1", "mic2", "mic3"]


def family(version: str) -> str:
    return "clean" if version == "clean" else ("pitch" if version.startswith("pitch") else "mic")


def change_pitch(x: np.ndarray, rate: int, steps: float) -> np.ndarray:
    """Shift pitch by ``steps`` semitones while keeping formants (Praat 'Change gender', PSOLA).

    A person saying a higher or lower 'aaah' moves their pitch but keeps the
    shape of their vocal tract, which a plain resampling pitch shift would not.
    """
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(x.astype(np.float64), sampling_frequency=rate)
    median = call(sound.to_pitch(pitch_floor=60.0, pitch_ceiling=600.0), "Get quantile", 0, 0, 0.5, "Hertz")
    if not np.isfinite(median) or median <= 0:
        return x.astype(np.float32)
    shifted = call(sound, "Change gender", 60.0, 600.0, 1.0, float(median * 2 ** (steps / 12)), 1.0, 1.0)
    return np.asarray(shifted.values[0], dtype=np.float32)


def mic_channel(x: np.ndarray, rate: int, rng: np.random.Generator, out_rate: int = 16_000) -> tuple[np.ndarray, int]:
    """Re-record a clean signal as a laptop microphone in a room would."""
    y = audio.resample(x, rate, out_rate)
    # Smooth random frequency response (about +-2.5 dB per band).
    freqs = [0, 300, 1000, 3000, out_rate / 2]
    gains = 10 ** (rng.normal(0, 2.5, len(freqs)) / 20)
    y = sps.lfilter(sps.firwin2(101, freqs, gains, fs=out_rate), 1.0, y).astype(np.float32)
    # Room reverberation: a short exponentially decaying noise tail.
    if rng.random() < 0.7:
        n = int(rng.uniform(0.1, 0.35) * out_rate)
        tail = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)
        tail *= np.sqrt(rng.uniform(0.05, 0.25) / np.sum(tail ** 2))
        y = y + sps.fftconvolve(y, tail)[: len(y)].astype(np.float32)
    # Silence before and after, with background noise (white plus low-frequency rumble) at 15-35 dB SNR.
    lead, trail = (int(rng.uniform(0.3, 1.0) * out_rate) for _ in range(2))
    full = np.concatenate([np.zeros(lead), y, np.zeros(trail)]).astype(np.float32)
    white = rng.standard_normal(len(full))
    rumble = sps.sosfilt(sps.butter(2, 300, fs=out_rate, output="sos"), white)
    noise = white + 2.0 * rumble
    noise *= (np.sqrt(np.mean(y ** 2)) / 10 ** (rng.uniform(15, 35) / 20)) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    full = (full + noise) * rng.uniform(0.05, 0.8)
    return full.astype(np.float32), out_rate


def make_version(raw: np.ndarray, rate: int, version: str, seed_key: str) -> tuple[np.ndarray, int]:
    """Deterministically perturb a raw recording; the same (key, version) always gives the same audio."""
    rng = np.random.default_rng(zlib.crc32(f"{seed_key}|{version}".encode()))
    if version == "clean":
        return raw, rate
    if version.startswith("pitch"):
        return change_pitch(raw, rate, float(version[5:])), rate
    if rng.random() < 0.5:
        raw = change_pitch(raw, rate, float(rng.uniform(-3, 3)))
    return mic_channel(raw, rate, rng)
