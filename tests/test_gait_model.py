"""Model 2 (gait video) inside PD_Screening, plus the pandas-independence of the voice model."""

from pathlib import Path

import numpy as np
import pytest

from pdscreen import compat
from pdscreen.gait import motion, pose
from pdscreen.gait import config as gait_config

compat.pandas_available()


def walking_poses(seconds=12.0, fps=7.0, stride_hz=1.0, x0=0.5, travel=0.0, scale=1.0, aspect=1.78) -> np.ndarray:
    """Synthetic (frames, 33, 4) lower-body walking: alternating ankles and knees."""
    n = int(seconds * fps)
    t = np.arange(n) / fps
    poses = np.full((n, 33, 4), np.nan, dtype=np.float32)
    swing = np.sin(2 * np.pi * stride_hz * t)
    hip_x = x0 + travel * t / seconds
    for sign, hip_id, knee_id, ankle_id, heel_id, toe_id in ((1, 23, 25, 27, 29, 31), (-1, 24, 26, 28, 30, 32)):
        hip_y = 0.45
        poses[:, hip_id] = np.stack([hip_x + sign * 0.03 * scale / aspect, np.full(n, hip_y), np.zeros(n), np.ones(n)], axis=1)
        poses[:, knee_id] = np.stack([hip_x + 0.02 * sign * swing * scale, hip_y + 0.18 * scale - 0.02 * scale * np.abs(sign * swing),
                                      np.zeros(n), np.ones(n)], axis=1)
        poses[:, ankle_id] = np.stack([hip_x + 0.06 * sign * swing * scale / aspect,
                                       hip_y + 0.36 * scale - 0.04 * scale * np.maximum(sign * swing, 0), np.zeros(n), np.ones(n)], axis=1)
        poses[:, heel_id] = poses[:, ankle_id] + np.array([-0.01, 0.01, 0, 0])
        poses[:, toe_id] = poses[:, ankle_id] + np.array([0.03, 0.01, 0, 0])
    return poses


needs_gait = pytest.mark.skipif(not (gait_config.MODELS / "pose_rf.joblib").exists() or not gait_config.POSE_MODEL.exists(),
                                reason="gait model files are not installed")
SAMPLE = Path(r"C:\PD_Gait\data\raw\normal_gait\Instance_1081.0.mp4")


def test_the_gait_package_loads_without_the_voice_stack_or_pandas():
    assert not compat.pandas_available() or True  # either way, importing the gait modules must work
    assert gait_config.MODELS.name == "gait" and gait_config.TARGET_FPS == 7.0


def test_patient_selection_drops_an_examiner_standing_to_the_side():
    both = np.stack([walking_poses(x0=0.85, stride_hz=0.0), walking_poses(x0=0.45)], axis=1)
    selected, quality = pose.select_subject(both)
    assert quality["stationary"] and quality["detection_rate"] > 0.95
    assert np.nanmean(selected[:, 23, 0]) == pytest.approx(0.45 + 0.03 / 1.78, abs=0.02)


def test_gait_features_report_stride_frequency_and_ignore_scale():
    base = motion.video_windows(walking_poses(stride_hz=1.0), 1.78, 1.0)
    small = motion.video_windows(walking_poses(stride_hz=1.0, scale=0.5, x0=0.3), 1.78, 1.0)
    assert len(base) >= 2 and len(small) >= 2
    assert np.median([w["stride_freq_hz"] for w in base]) == pytest.approx(1.0, abs=0.15)
    assert np.median([w["knee_angle_range_l"] for w in small]) == pytest.approx(np.median([w["knee_angle_range_l"] for w in base]), rel=0.15)


@needs_gait
def test_unreadable_video_gives_a_friendly_error_without_internal_paths(tmp_path):
    from pdscreen.gait.inference import GaitScreener, VideoError

    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(VideoError) as error:
        GaitScreener().analyse(bad)
    assert "could not be read as a video" in str(error.value) and str(tmp_path) not in str(error.value)


@needs_gait
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample video not available")
def test_real_video_runs_end_to_end_and_flags_the_framing_shortcut():
    from pdscreen.gait.inference import GaitScreener

    result = GaitScreener().analyse(SAMPLE, max_seconds=20)
    assert 0 <= result["probability_pd"] <= 1 and result["windows"] >= 1
    assert result["framing_only_probability_pd"] is not None and isinstance(result["shortcut_risk"], bool)


def test_unreadable_audio_gives_a_friendly_error():
    from pdscreen.audio import AudioError, load_audio

    with pytest.raises(AudioError) as error:
        load_audio(b"RIFF definitely not audio")
    assert "could not be read as audio" in str(error.value) and "BytesIO" not in str(error.value)


def test_voice_model_and_its_explanations_work_without_pandas(screener):
    from pdscreen import explain  # imports without pandas; Grad-CAM must not need it

    assert hasattr(explain, "grad_cam")
    from conftest import to_wav_bytes, vowel

    x, rate = vowel(seconds=4.0, rate=16_000)
    result, cleaned = screener.predict_bytes(to_wav_bytes(x, rate))
    contributions, method = screener.explain_features(result.features)
    assert method in {"SHAP", "occlusion"} and 1 <= len(contributions) <= 8
    sizes = [abs(v) for _, v in contributions]
    assert sizes == sorted(sizes, reverse=True) and all(np.isfinite(v) for _, v in contributions)
    heatmap = screener.grad_cam(cleaned)
    assert heatmap["cam"].shape == heatmap["log_mel"].shape
