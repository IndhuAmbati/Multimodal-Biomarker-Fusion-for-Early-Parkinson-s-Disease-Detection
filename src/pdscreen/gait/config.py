"""Constants for the gait-video model (Model 2). Paths are inside this project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models" / "gait"
POSE_MODEL = MODELS / "pose_landmarker_full.task"

SEED = 42
TARGET_FPS = 7.0          # every video is resampled to this rate (the WeightGait videos are already 7 fps)
MAX_WIDTH = 640           # frames are shrunk to this width before pose estimation
NUM_POSES = 2             # the PD videos show an examiner next to the patient, so look for two people

# MediaPipe pose landmark indices used for gait. The PD videos are cropped below the neck, so only the
# lower body (hips down) is used for every video, which keeps the two sources comparable.
HIPS = (23, 24)
KNEES = (25, 26)
ANKLES = (27, 28)
HEELS = (29, 30)
FOOT_INDEX = (31, 32)
LOWER_BODY = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
SHOULDERS = (11, 12)
HEAD = [0, 1, 2, 3, 4, 5, 6, 7, 8]

WINDOW_SECONDS = 3.0      # the same for every video: the normal-walking clips show the person for only about 3 s
WINDOW_HOP_SECONDS = 1.5
MIN_WINDOW_DETECTION = 0.6  # share of a window's frames in which the person must be found

DISCLAIMER = "Research prototype. Not a diagnosis. See the shortcut audit before reading any score."
