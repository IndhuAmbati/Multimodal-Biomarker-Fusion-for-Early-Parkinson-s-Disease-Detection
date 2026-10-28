"""Reproduce everything: data preparation, all models, reports and the deployed model.

    python scripts/train_all.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["prepare_data.py", "train_classic.py", "train_cnn.py", "evaluate_uci.py", "prepare_windows.py",
         "baseline_robustness.py", "train_robust.py", "train_voice_id.py", "finalize.py", "benchmark_latency.py"]

if __name__ == "__main__":
    for step in STEPS:
        start = time.perf_counter()
        print(f"\n=== {step} ===", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts" / step)], check=True, cwd=ROOT)
        print(f"=== {step} finished in {time.perf_counter() - start:.0f} s ===", flush=True)
