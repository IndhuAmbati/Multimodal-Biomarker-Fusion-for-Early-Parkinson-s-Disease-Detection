"""Paths and constants shared by training, evaluation and inference."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FIGSHARE_DIR = DATA_RAW / "figshare_iyer2023"
UCI_DIR = DATA_RAW / "uci_parkinsons"

# Every recording, in training and live use, is converted to this rate. The
# training data are 8 kHz telephone recordings, so live microphone audio is
# band-limited the same way to keep both on one acoustic footing.
SAMPLE_RATE = 8_000
BANDPASS_HZ = (80.0, 3_400.0)
MIN_VOICED_SECONDS = 1.0
MAX_INPUT_SECONDS = 30.0

# The feature model scores fixed-length windows (1.5 s is the length Iyer et al. analysed) and takes the median,
# so a 3 s and a 10 s recording are judged on the same footing and onset/offset transients cannot dominate.
ANALYSIS_WINDOW_SECONDS = 1.5
ANALYSIS_HOP_SECONDS = 0.5
INFERENCE_MAX_WINDOWS = 8
WINDOW_TARGET_RMS = 0.1

# Log-Mel spectrogram windows for the CNN.
WINDOW_SECONDS = 1.0
WINDOW_HOP_SECONDS = 0.5
N_FFT = 256
HOP_LENGTH = 80
N_MELS = 64

SEED = 42
LABELS = {"HC": 0, "PD": 1}
DISCLAIMER = "Screening aid only. Consult a neurologist."
