import time

import numpy as np
import pytest
from conftest import to_wav_bytes, vowel

from pdscreen import config
from pdscreen.audio import preprocess


def held_out_wav() -> bytes:
    import csv

    with open(config.DATA_PROCESSED / "figshare_features.csv", newline="") as f:
        row = next(r for r in csv.DictReader(f) if r["test"] == "True")
    return open(row["path"], "rb").read()


def test_prediction_is_valid_deterministic_and_fast(screener, vowel_wav):
    first, _ = screener.predict_bytes(vowel_wav)
    second, _ = screener.predict_bytes(vowel_wav)
    assert 0.0 <= first.probability <= 1.0
    assert first.probability == pytest.approx(second.probability)
    assert first.label == ("PD likely" if first.probability >= screener.threshold else "Healthy")
    assert first.confidence in {"High", "Moderate", "Low"}

    x, rate = vowel(seconds=10.0, rate=16_000)
    start = time.perf_counter()
    result, _ = screener.predict_bytes(to_wav_bytes(x, rate))
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"end-to-end latency {elapsed:.2f} s exceeds the 2 s budget"
    assert result.latency_ms["total"] < 2000


def test_held_out_recording_runs_end_to_end(screener):
    result, cleaned = screener.predict_bytes(held_out_wav())
    payload = result.as_dict()
    assert payload["status"] == "completed" and payload["disclaimer"] == config.DISCLAIMER
    assert set(payload["components"]) == set(screener.components)
    assert len(cleaned) > config.SAMPLE_RATE


def test_onnx_matches_pytorch(screener):
    if screener.session is None:
        pytest.skip("CNN not deployed")
    import torch

    from pdscreen.cnn import MelCNN, windows

    x, rate = vowel(seconds=3.0)
    y, _ = preprocess(x, rate)
    model = MelCNN()
    model.load_state_dict(torch.load(config.MODELS / "mel_cnn.pt", map_location="cpu"))
    model.eval()
    with torch.no_grad():
        mel = screener.front(torch.tensor(windows(y)))[:, None]
        expected = model(mel).numpy()
    actual = screener.session.run(None, {"log_mel": mel.numpy()})[0]
    assert np.allclose(expected, actual, atol=1e-4)


def test_api_predict_and_errors(screener, vowel_wav):
    from fastapi.testclient import TestClient

    from pdscreen.api import app

    with TestClient(app) as client:
        assert client.get("/health").json()["ready"] is True
        ok = client.post("/predict", files={"file": ("aaah.wav", vowel_wav, "audio/wav")})
        assert ok.status_code == 200
        body = ok.json()
        assert body["label"] in {"PD likely", "Healthy"} and 0 <= body["probability_pd"] <= 1
        assert client.post("/predict", files={"file": ("bad.wav", b"garbage", "audio/wav")}).status_code == 422
        assert client.post("/predict", files={"file": ("empty.wav", b"", "audio/wav")}).status_code == 400
