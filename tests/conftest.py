import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def vowel(seconds=4.0, rate=16_000, f0=140.0, jitter=0.0, noise=0.003, pad=0.5, seed=0):
    """Synthetic sustained vowel: harmonic glottal-like source with optional period jitter."""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    periods = 1.0 / (f0 * (1 + jitter * rng.standard_normal(int(seconds * f0 * 2))))
    phase_times = np.cumsum(periods)
    t = np.arange(n) / rate
    cycle = np.searchsorted(phase_times, t)
    start = np.concatenate([[0.0], phase_times])[np.minimum(cycle, len(phase_times) - 1)]
    local = (t - start) / periods[np.minimum(cycle, len(periods) - 1)]
    x = sum((0.6 / k) * np.sin(2 * np.pi * k * local) for k in range(1, 8))
    x = 0.3 * x / np.max(np.abs(x)) + noise * rng.standard_normal(n)
    silence = noise * rng.standard_normal(int(pad * rate))
    return np.concatenate([silence, x, silence]).astype(np.float32), rate


def to_wav_bytes(x, rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.fixture
def vowel_wav():
    x, rate = vowel()
    return to_wav_bytes(x, rate)


@pytest.fixture(scope="session")
def screener():
    from pdscreen import config

    if not (config.MODELS / "deployment.json").exists():
        pytest.skip("models not trained yet")
    from pdscreen.inference import Screener

    s = Screener()
    s.warm_up()
    return s
