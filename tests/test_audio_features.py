import numpy as np
import pytest
from conftest import to_wav_bytes, vowel

from pdscreen import audio, config, features


def test_preprocess_resamples_trims_and_reports_quality():
    x, rate = vowel(seconds=4.0, rate=16_000, pad=0.6)
    y, quality = audio.preprocess(x, rate)
    assert quality.original_sample_rate == 16_000
    assert 3.6 < len(y) / config.SAMPLE_RATE < 4.4  # padding trimmed, vowel kept
    assert np.max(np.abs(y)) == pytest.approx(0.95, abs=1e-3)
    assert np.isfinite(quality.snr_db) and quality.snr_db > 15


def test_short_silent_and_invalid_audio_are_rejected():
    short, rate = vowel(seconds=0.4, pad=0.2)
    with pytest.raises(audio.AudioError):
        audio.preprocess(short, rate)
    with pytest.raises(audio.AudioError):
        audio.preprocess(np.zeros(16_000 * 3, dtype=np.float32), 16_000)
    with pytest.raises(audio.AudioError):
        audio.load_audio(b"not a wav file")


def test_load_audio_reads_wav_bytes():
    x, rate = vowel(seconds=2.0)
    signal, read_rate = audio.load_audio(to_wav_bytes(x, rate))
    assert read_rate == rate and len(signal) == len(x)


def test_feature_vector_is_complete_and_finite():
    x, rate = vowel(seconds=3.0)
    y, _ = audio.preprocess(x, rate)
    values = features.extract_features(y)
    assert list(values) == features.FEATURE_NAMES
    assert all(np.isfinite(v) for v in values.values())
    assert values["f0_mean_hz"] == pytest.approx(140, rel=0.05)


def test_jitter_rises_with_period_irregularity():
    steady, rate = vowel(seconds=3.0, jitter=0.0)
    shaky, _ = vowel(seconds=3.0, jitter=0.02)
    j_steady = features.extract_features(audio.preprocess(steady, rate)[0])["jitter_local"]
    j_shaky = features.extract_features(audio.preprocess(shaky, rate)[0])["jitter_local"]
    assert j_shaky > 3 * j_steady


def test_rpde_and_dfa_behave_as_expected():
    rng = np.random.default_rng(1)
    t = np.arange(8_000) / 8_000
    periodic = np.sin(2 * np.pi * 125 * t)
    noise = rng.standard_normal(8_000)
    assert features.rpde(periodic) < features.rpde(noise)
    assert 0.35 < features.dfa(noise) < 0.65  # white noise has a DFA exponent of about 0.5
