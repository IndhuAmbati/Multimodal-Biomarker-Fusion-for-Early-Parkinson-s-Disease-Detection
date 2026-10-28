"""Measure end-to-end latency (audio bytes in -> result out) on held-out recordings.

    python scripts/benchmark_latency.py
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pdscreen import config  # noqa: E402
from pdscreen.inference import Screener  # noqa: E402


def as_wav(x: np.ndarray, rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, x, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def main(runs: int = 5) -> None:
    start = time.perf_counter()
    screener = Screener()
    load = time.perf_counter() - start
    start = time.perf_counter()
    screener.warm_up()
    warm = time.perf_counter() - start

    table = pd.read_csv(config.DATA_PROCESSED / "figshare_features.csv")
    paths = table[table.test].path.tolist()
    results = {"model_load_seconds": round(load, 2), "warm_up_seconds": round(warm, 2)}
    for seconds in (5, 10):
        totals, stages = [], []
        for path in paths:
            x, rate = sf.read(path, dtype="float32")
            # Loop the recording to the target length and upsample to 16 kHz like the browser recorder.
            x = np.tile(x, int(np.ceil(seconds * rate / len(x))))[: seconds * rate]
            x16 = np.interp(np.arange(seconds * 16_000) / 2, np.arange(len(x)), x).astype(np.float32)
            payload = as_wav(x16, 16_000)
            for _ in range(runs):
                t0 = time.perf_counter()
                result, _ = screener.predict_bytes(payload)
                totals.append((time.perf_counter() - t0) * 1000)
                stages.append(result.latency_ms)
        frame = pd.DataFrame(stages)
        results[f"{seconds}s_audio"] = {
            "runs": len(totals),
            "end_to_end_ms_median": round(float(np.median(totals)), 1),
            "end_to_end_ms_p95": round(float(np.percentile(totals, 95)), 1),
            "end_to_end_ms_max": round(float(np.max(totals)), 1),
            "stage_ms_median": {k: round(float(v), 1) for k, v in frame.median().items()},
        }
    (config.REPORTS / "latency.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
