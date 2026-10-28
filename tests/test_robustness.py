import numpy as np
import pytest
from conftest import vowel

from pdscreen import audio, augment, config


def test_analysis_windows_are_equal_length_and_level():
    rng = np.random.default_rng(0)
    y = (rng.standard_normal(config.SAMPLE_RATE * 9) * np.linspace(0.2, 1.0, config.SAMPLE_RATE * 9)).astype(np.float32)
    windows = audio.analysis_windows(y)
    size = int(config.ANALYSIS_WINDOW_SECONDS * config.SAMPLE_RATE)
    assert len(windows) > 8 and all(len(w) == size for w in windows)
    rms = [float(np.sqrt(np.mean(w ** 2))) for w in windows]
    assert max(rms) - min(rms) < 1e-3  # loudness drift does not change what the windows look like
    capped = audio.analysis_windows(y, max_windows=config.INFERENCE_MAX_WINDOWS)
    assert len(capped) == config.INFERENCE_MAX_WINDOWS


def test_short_recording_gives_one_window():
    assert len(audio.analysis_windows(np.ones(config.SAMPLE_RATE, dtype=np.float32) * 0.1)) == 1


def test_perturbations_are_deterministic_and_change_the_audio():
    x, rate = vowel(seconds=3.0, rate=8_000)
    a, ra = augment.make_version(x, rate, "mic1", "person")
    b, rb = augment.make_version(x, rate, "mic1", "person")
    assert ra == rb == 16_000 and np.array_equal(a, b)
    assert len(a) > len(x) * 2 * 0.99  # upsampled to 16 kHz with silence added around the vowel
    same, _ = augment.make_version(x, rate, "clean", "person")
    assert np.array_equal(same, x)


def test_pitch_change_moves_pitch_by_the_requested_interval():
    import parselmouth

    x, rate = vowel(seconds=3.0, rate=8_000, f0=120.0)

    def median_f0(signal):
        pitch = parselmouth.Sound(signal.astype(np.float64), sampling_frequency=rate).to_pitch(pitch_floor=60, pitch_ceiling=600)
        f0 = pitch.selected_array["frequency"]
        return float(np.median(f0[f0 > 0]))

    base = median_f0(x)
    up, down = median_f0(augment.change_pitch(x, rate, 4.0)), median_f0(augment.change_pitch(x, rate, -4.0))
    assert up == pytest.approx(base * 2 ** (4 / 12), rel=0.06)
    assert down == pytest.approx(base * 2 ** (-4 / 12), rel=0.06)


def test_preprocess_handles_a_simulated_microphone_recording():
    x, rate = vowel(seconds=4.0, rate=8_000)
    mic, mic_rate = augment.mic_channel(x, rate, np.random.default_rng(3))
    y, quality = audio.preprocess(mic, mic_rate)
    assert quality.voiced_seconds > 2.5 and np.max(np.abs(y)) == pytest.approx(0.95, abs=1e-3)
