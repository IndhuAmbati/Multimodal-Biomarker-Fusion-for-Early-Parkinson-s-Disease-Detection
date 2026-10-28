"""MediaPipe pose extraction and selection of the walking person.

Every video is read at TARGET_FPS (by timestamp, so a 30 fps and a 7 fps video give the same time spacing), shrunk,
and passed to MediaPipe's PoseLandmarker. Up to two people are kept per frame; ``select_subject`` then picks the
walking person and drops an examiner standing to the side.

Output of ``extract_pose``: ``poses`` has shape (frames, people, 33, 4) holding x, y, z and visibility, with NaN where
no pose was found. x and y are in image-normalised units (0 to 1).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config

N_LANDMARKS = 33


def create_landmarker(model_path: Path = config.POSE_MODEL, num_poses: int = config.NUM_POSES):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=num_poses,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3, min_tracking_confidence=0.3)
    return vision.PoseLandmarker.create_from_options(options), mp


def extract_pose(video_path: str | Path, target_fps: float = config.TARGET_FPS, max_width: int = config.MAX_WIDTH,
                 model_path: Path = config.POSE_MODEL, max_seconds: float | None = None, progress=None) -> dict:
    """Run pose estimation on a video; returns poses, timestamps and video metadata.

    ``progress`` is an optional callable that receives the fraction of the video processed so far (0 to 1).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_seconds = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / src_fps
    if max_seconds is not None and total_seconds:
        total_seconds = min(total_seconds, max_seconds)
    landmarker, mp = create_landmarker(model_path)
    times, poses, thumbs = [], [], []
    next_time, index, last_ms = 0.0, 0, -1
    try:
        while True:
            if not cap.grab():
                break
            t = index / src_fps
            index += 1
            if max_seconds is not None and t > max_seconds:
                break
            if t + 1e-6 < next_time:
                continue
            next_time += 1.0 / target_fps
            ok, frame = cap.retrieve()
            if not ok:
                break
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(round(frame.shape[0] * scale))))
            last_ms = max(int(round(t * 1000)), last_ms + 1)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), last_ms)
            frame_poses = np.full((config.NUM_POSES, N_LANDMARKS, 4), np.nan, dtype=np.float32)
            for k, landmarks in enumerate(result.pose_landmarks[: config.NUM_POSES]):
                frame_poses[k] = [(l.x, l.y, l.z, l.visibility) for l in landmarks]
            times.append(t)
            poses.append(frame_poses)
            if progress is not None and total_seconds and len(times) % 10 == 0:
                progress(min(t / total_seconds, 1.0))
            if len(thumbs) < 400 and len(times) % 12 == 1:
                thumbs.append(cv2.resize(frame, (160, int(frame.shape[0] * 160 / frame.shape[1]))))
    finally:
        cap.release()
        landmarker.close()
    if not poses:
        raise ValueError(f"No frames could be read from {video_path}")
    return {"t": np.asarray(times, dtype=np.float32), "poses": np.stack(poses), "src_fps": float(src_fps),
            "width": width, "height": height, "aspect": width / height}


def _centre(pose: np.ndarray, min_visibility: float = 0.3) -> np.ndarray | None:
    """Hip centre of one pose, or the mean of its visible landmarks if the hips are not visible."""
    hips = pose[list(config.HIPS)]
    if np.all(np.isfinite(hips[:, :2])) and np.all(hips[:, 3] > min_visibility):
        return hips[:, :2].mean(axis=0)
    visible = pose[np.isfinite(pose[:, 0]) & (pose[:, 3] > min_visibility)]
    return visible[:, :2].mean(axis=0) if len(visible) >= 4 else None


def candidates(poses: np.ndarray) -> list[list[tuple[int, np.ndarray]]]:
    """For every frame, the usable people as (pose index, centre) pairs."""
    out = []
    for f in range(poses.shape[0]):
        found = []
        for p in range(poses.shape[1]):
            if np.isfinite(poses[f, p, :, 0]).any():
                centre = _centre(poses[f, p])
                if centre is not None:
                    found.append((p, centre))
        out.append(found)
    return out


def _leg_motion(pose_array: np.ndarray) -> float:
    """How much the ankles move relative to the hips (in hip-to-ankle lengths) over a set of poses."""
    hips = pose_array[:, list(config.HIPS), :2].mean(axis=1)
    ankles = pose_array[:, list(config.ANKLES), :2].mean(axis=1)
    length = np.nanmedian(np.linalg.norm(ankles - hips, axis=1))
    if not np.isfinite(length) or length < 1e-3 or len(pose_array) < 5:
        return 0.0
    return float(np.nanstd((ankles - hips)[:, 1]) / length)


def select_subject(poses: np.ndarray, bin_width: float = 0.1, tolerance: float = 0.25) -> tuple[np.ndarray, dict]:
    """Choose the walking person in every frame. Returns (frames, 33, 4) poses and a small quality report.

    If someone is detected around one place in the image for most of the video (a person turning or stepping on the
    spot, with an examiner standing to the side), only people near that place are kept. Among the most common
    places, the one whose legs move most is the patient. If detections spread across the image (a person walking
    across the scene), every detection is kept. Within a frame the person closest to the previous choice wins.
    """
    per_frame = candidates(poses)
    out = np.full((poses.shape[0], N_LANDMARKS, 4), np.nan, dtype=np.float32)
    entries = [(f, p, c) for f, frame in enumerate(per_frame) for p, c in frame]
    if not entries:
        return out, {"detection_rate": 0.0, "stationary": False, "frames_with_other_people": 0}
    xs = np.array([c[0] for _, _, c in entries])
    counts, edges = np.histogram(xs, bins=np.arange(0.0, 1.0 + bin_width, bin_width))
    eligible = [i for i in range(len(counts)) if counts[i] >= 0.6 * counts.max()]

    def bin_motion(i: int) -> float:
        members = [poses[f, p] for f, p, c in entries if edges[i] <= c[0] < edges[i + 1] + 1e-9]
        return _leg_motion(np.stack(members)) if members else 0.0

    best = max(eligible, key=bin_motion)
    mode = float(edges[best] + bin_width / 2)
    frames_with_nearby = sum(any(abs(c[0] - mode) <= tolerance for _, c in frame) for frame in per_frame if frame)
    stationary = bool(frames_with_nearby / max(1, sum(1 for frame in per_frame if frame)) >= 0.7)
    anchor = float(np.median([c[0] for _, _, c in entries if abs(c[0] - mode) <= tolerance])) if stationary else None
    previous, chosen, others = None, 0, 0
    for f, frame in enumerate(per_frame):
        pool = [(p, c) for p, c in frame if not stationary or abs(c[0] - anchor) <= tolerance]
        others += int(len(pool) < len(frame))
        if not pool:
            continue
        reference_x = previous[0] if previous is not None else anchor
        p, c = min(pool, key=lambda pc: 0.0 if reference_x is None else abs(pc[1][0] - reference_x))
        out[f] = poses[f, p]
        previous, chosen = c, chosen + 1
    return out, {"detection_rate": chosen / poses.shape[0], "stationary": stationary, "frames_with_other_people": others}


def save_pose(path: Path, extracted: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in extracted.items()})


def load_pose(path: Path) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
