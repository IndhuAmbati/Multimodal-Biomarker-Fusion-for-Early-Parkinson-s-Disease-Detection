"""Gait features from a pose sequence.

All lengths are divided by the person's own leg length and all rates are per second, so a feature does not
depend on camera distance, image resolution or frame rate. Coordinates are first made isotropic by multiplying x
by the image aspect ratio.

Features are grouped so their role can be audited:
- GAIT_FEATURES: movement of the legs themselves (cadence, regularity, knee angle, foot clearance, symmetry, sway).
- TASK_FEATURES: how the person moves through the scene (net travel, path directness, standing still). These
  reflect the walking task (walking across a room vs turning on the spot), not the walking pattern.
- NUISANCE_FEATURES: facts about the camera, framing and pose detector. They carry no gait information.
"""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import config

GAIT_FEATURES = [
    "stride_freq_hz", "step_regularity", "step_time_cv", "spectral_entropy", "ankle_vsep_range", "ankle_hsep_range",
    "knee_angle_range_l", "knee_angle_range_r", "knee_angle_asym", "knee_angle_mean", "foot_clearance_l",
    "foot_clearance_r", "foot_clearance_asym", "hip_sway", "hip_bounce", "ankle_speed_mean", "ankle_speed_p95",
    "ankle_accel_std", "foot_angle_range", "leg_motion_asym",
]
TASK_FEATURES = ["net_speed", "path_speed", "path_directness", "rest_fraction"]
NUISANCE_FEATURES = ["head_in_frame", "shoulders_in_frame", "bbox_height", "bbox_centre_x", "bbox_centre_y",
                     "detection_rate", "lower_visibility"]
FEATURE_GROUPS = {"gait": GAIT_FEATURES, "task": TASK_FEATURES, "nuisance": NUISANCE_FEATURES}
ALL_FEATURES = GAIT_FEATURES + TASK_FEATURES + NUISANCE_FEATURES


def present(poses: np.ndarray) -> np.ndarray:
    """Boolean per frame: was a person found (any landmark with a position)."""
    return np.isfinite(poses[:, :, 0]).any(axis=1)


def _interpolate(x: np.ndarray, max_gap: int = 4) -> np.ndarray:
    """Fill short runs of NaN (per column) by linear interpolation; longer gaps stay NaN."""
    x = x.copy()
    for c in range(x.shape[1]):
        col = x[:, c]
        bad = ~np.isfinite(col)
        if not bad.any() or bad.all():
            continue
        idx = np.arange(len(col))
        filled = np.interp(idx, idx[~bad], col[~bad])
        run = 0
        for i in range(len(col)):
            run = run + 1 if bad[i] else 0
            if bad[i] and run > max_gap:
                for j in range(i - run + 1, i + 1):
                    filled[j] = np.nan
        x[:, c] = filled
    return x


def prepare(poses: np.ndarray, aspect: float, fps: float = config.TARGET_FPS, min_visibility: float = 0.3) -> dict:
    """Body-centred, leg-length-normalised lower-body trajectories.

    ``poses`` is (frames, 33, 4) for one person. Returns xy (frames, 33, 2) in leg-length units relative to the hip
    centre, hip (frames, 2) in leg-length units in the scene, the leg length, and a per-frame validity mask.
    """
    xy = poses[:, :, :2].astype(np.float64).copy()
    xy[:, :, 0] *= aspect
    visibility = poses[:, :, 3]
    lower = list(config.LOWER_BODY)
    unusable = np.zeros(xy.shape[:2], dtype=bool)
    unusable[:, lower] = ~np.isfinite(xy[:, lower, 0]) | (visibility[:, lower] < min_visibility)
    xy[unusable] = np.nan
    flat = _interpolate(xy.reshape(len(xy), -1)).reshape(xy.shape)
    for j in lower:  # light smoothing of each coordinate over time
        for d in range(2):
            col = flat[:, j, d]
            ok = np.isfinite(col)
            if ok.sum() >= 7:
                filled = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
                flat[:, j, d] = np.where(ok, signal.savgol_filter(filled, 5, 2), np.nan)
    hip = flat[:, list(config.HIPS), :].mean(axis=1)
    ankle = flat[:, list(config.ANKLES), :].mean(axis=1)
    leg = np.nanmedian(np.linalg.norm(ankle - hip, axis=1))
    if not np.isfinite(leg) or leg < 1e-3:
        raise ValueError("Legs are not visible enough to measure gait.")
    valid = np.all(np.isfinite(flat[:, lower, :]).reshape(len(flat), -1), axis=1)
    return {"xy": (flat - hip[:, None, :]) / leg, "hip": hip / leg, "leg": float(leg), "valid": valid, "fps": fps}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b (degrees) between the segments b->a and b->c."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _range(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.percentile(x, 95) - np.percentile(x, 5)) if len(x) > 3 else np.nan


def _stride_stats(sep: np.ndarray, fps: float) -> tuple[float, float, float, float]:
    """Stride frequency, regularity, step-time variability and spectral entropy of the ankle separation signal."""
    s = sep[np.isfinite(sep)]
    if len(s) < int(2 * fps):
        return (np.nan,) * 4
    s = signal.detrend(s)
    freqs, power = signal.periodogram(s, fs=fps, nfft=max(256, 4 * len(s)))
    band = (freqs >= 0.4) & (freqs <= min(3.2, fps / 2 - 0.2))
    if not band.any() or power[band].sum() <= 0:
        return (np.nan,) * 4
    stride = float(freqs[band][np.argmax(power[band])])
    p = power[band] / power[band].sum()
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    ac = signal.correlate(s, s, mode="full")[len(s) - 1:]
    ac = ac / (ac[0] + 1e-12)
    lag = int(round(fps / stride))
    lo, hi = max(1, int(lag * 0.7)), min(len(ac) - 1, int(lag * 1.3) + 1)
    regularity = float(ac[lo:hi + 1].max()) if hi > lo else np.nan
    peaks, _ = signal.find_peaks(s, distance=max(2, int(fps / 3.5)), prominence=0.25 * s.std())
    intervals = np.diff(peaks) / fps
    cv = float(intervals.std() / intervals.mean()) if len(intervals) >= 3 else np.nan
    return stride, regularity, cv, entropy


def window_features(prepared: dict, start: int, stop: int) -> dict[str, float]:
    """Gait and task features for frames [start, stop)."""
    xy, hip, fps = prepared["xy"][start:stop], prepared["hip"][start:stop], prepared["fps"]
    la, ra = xy[:, 27], xy[:, 28]
    out: dict[str, float] = {}
    stride, regularity, cv, entropy = _stride_stats(la[:, 1] - ra[:, 1], fps)
    out.update(stride_freq_hz=stride, step_regularity=regularity, step_time_cv=cv, spectral_entropy=entropy,
               ankle_vsep_range=_range(la[:, 1] - ra[:, 1]), ankle_hsep_range=_range(la[:, 0] - ra[:, 0]))
    knee_l = _angle(xy[:, 23], xy[:, 25], xy[:, 27])
    knee_r = _angle(xy[:, 24], xy[:, 26], xy[:, 28])
    out.update(knee_angle_range_l=_range(knee_l), knee_angle_range_r=_range(knee_r),
               knee_angle_asym=abs(_range(knee_l) - _range(knee_r)), knee_angle_mean=float(np.nanmean(np.r_[knee_l, knee_r])))
    clear_l, clear_r = _range(la[:, 1]), _range(ra[:, 1])
    out.update(foot_clearance_l=clear_l, foot_clearance_r=clear_r, foot_clearance_asym=abs(clear_l - clear_r))
    h = hip[np.isfinite(hip[:, 0])]
    if len(h) >= 5:
        t = np.arange(len(h))
        out["hip_sway"] = float(np.std(h[:, 0] - np.polyval(np.polyfit(t, h[:, 0], 1), t)))
        out["hip_bounce"] = float(np.std(h[:, 1] - np.polyval(np.polyfit(t, h[:, 1], 1), t)))
    else:
        out.update(hip_sway=np.nan, hip_bounce=np.nan)
    speeds = np.stack([np.linalg.norm(np.diff(a, axis=0), axis=1) * fps for a in (la, ra)])
    out.update(ankle_speed_mean=float(np.nanmean(speeds)), ankle_speed_p95=float(np.nanpercentile(speeds, 95)))
    accel = np.diff(speeds, axis=1) * fps
    out["ankle_accel_std"] = float(np.nanstd(accel))
    # Foot pitch relative to horizontal, in (-90, 90): no wrap-around when the foot points left or right.
    foot = np.degrees(np.arctan2(xy[:, [29, 30], 1] - xy[:, [31, 32], 1],
                                 np.abs(xy[:, [29, 30], 0] - xy[:, [31, 32], 0])))
    out["foot_angle_range"] = float(np.nanmean([_range(foot[:, 0]), _range(foot[:, 1])]))
    out["leg_motion_asym"] = float(abs(np.nanstd(la) - np.nanstd(ra)))
    duration = max((stop - start - 1) / fps, 1e-6)
    steps = np.linalg.norm(np.diff(h, axis=0), axis=1) if len(h) > 1 else np.array([0.0])
    path = float(steps.sum())
    net = float(np.linalg.norm(h[-1] - h[0])) if len(h) > 1 else 0.0
    out.update(net_speed=net / duration, path_speed=path / duration, path_directness=net / (path + 1e-6),
               rest_fraction=float(np.mean(np.all(speeds < 0.15, axis=0))))
    return out


def nuisance_features(poses: np.ndarray, detection_rate: float, start: int, stop: int) -> dict[str, float]:
    """Camera, framing and detector facts for frames [start, stop). Not gait."""
    p = poses[start:stop]
    found = present(p)
    if not found.any():
        return {k: np.nan for k in NUISANCE_FEATURES}
    q = p[found]
    inside = (q[:, :, 0] >= 0) & (q[:, :, 0] <= 1) & (q[:, :, 1] >= 0) & (q[:, :, 1] <= 1)
    # MediaPipe reports high confidence for landmarks it places outside the image, so use position, not confidence.
    y = np.clip(q[:, :, 1], 0, 1)
    return {"head_in_frame": float(inside[:, config.HEAD].mean()),
            "shoulders_in_frame": float(inside[:, list(config.SHOULDERS)].mean()),
            "bbox_height": float(np.median(y.max(axis=1) - y.min(axis=1))),
            "bbox_centre_x": float(np.clip(q[:, config.LOWER_BODY, 0], 0, 1).mean()),
            "bbox_centre_y": float(y[:, config.LOWER_BODY].mean()), "detection_rate": detection_rate,
            "lower_visibility": float(np.nanmean(q[:, config.LOWER_BODY, 3]))}


def video_windows(poses: np.ndarray, aspect: float, detection_rate: float, fps: float = config.TARGET_FPS,
                  seconds: float = config.WINDOW_SECONDS, hop: float = config.WINDOW_HOP_SECONDS) -> list[dict]:
    """Feature dictionaries for every window of the video in which the person was found often enough."""
    if not present(poses).any():
        return []
    prepared = prepare(poses, aspect, fps)
    size, step = int(round(seconds * fps)), int(round(hop * fps))
    found = present(poses)
    windows = []
    for start in range(0, len(poses) - size + 1, step):
        stop = start + size
        if found[start:stop].mean() < config.MIN_WINDOW_DETECTION:
            continue
        values = window_features(prepared, start, stop)
        values.update(nuisance_features(poses, float(found[start:stop].mean()), start, stop))
        values["start"], values["stop"] = start, stop
        windows.append(values)
    return windows


def feature_matrix(windows: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[w.get(n, np.nan) for n in names] for w in windows], dtype=np.float64).reshape(len(windows), len(names))
