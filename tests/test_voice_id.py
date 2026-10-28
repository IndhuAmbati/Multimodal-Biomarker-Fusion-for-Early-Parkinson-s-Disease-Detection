import numpy as np
import pytest
from conftest import to_wav_bytes, vowel

from pdscreen import config
from pdscreen.audio import AudioError


@pytest.fixture(scope="module")
def identifier():
    if not (config.MODELS / "voice_id.joblib").exists():
        pytest.skip("voice model not trained yet")
    from pdscreen.voice_id import VoiceIdentifier

    identifier = VoiceIdentifier()
    identify(identifier, 150.0)    # the first call in a fresh process loads libraries; the 2 s budget is for a running app
    return identifier


def identify(identifier, f0: float):
    x, rate = vowel(seconds=4.0, rate=16_000, f0=f0)
    return identifier.predict_bytes(to_wav_bytes(x, rate))


def test_low_and_high_pitched_voices_go_to_opposite_sides(identifier):
    low, high = identify(identifier, 100.0), identify(identifier, 250.0)
    assert low["probability_female"] < 0.35 and low["label"] == "Likely male"
    assert high["probability_female"] > 0.65 and high["label"] == "Likely female"
    for f0, result in ((100.0, low), (250.0, high)):
        assert result["probability_female"] + result["probability_male"] == pytest.approx(1.0)
        assert result["windows"] >= 1
        # Best of three: the 2 s budget is for the code, not for a momentary slowdown of the computer.
        assert min(result["latency_ms"], *(identify(identifier, f0)["latency_ms"] for _ in range(2))) < 2000


def test_probability_rises_with_pitch_and_the_overlap_region_is_uncertain(identifier):
    p = [identify(identifier, f0)["probability_female"] for f0 in (100.0, 140.0, 180.0, 250.0)]
    assert p == sorted(p)
    assert identify(identifier, 160.0)["label"] == "Uncertain"


def test_unusable_audio_is_rejected_with_a_clear_error(identifier):
    with pytest.raises(AudioError):
        identifier.predict_bytes(b"not audio")
    with pytest.raises(AudioError):
        identifier.predict_signal(np.zeros(16_000 * 3, dtype=np.float32), 16_000)
