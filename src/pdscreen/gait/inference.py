"""Video in, prediction out: pose extraction -> gait features -> classifier -> probability and warnings."""

from __future__ import annotations

from pathlib import Path

import cv2
import joblib
import numpy as np

from . import config, motion, pose


class VideoError(ValueError):
    """The video cannot be analysed (unreadable, no person, legs not visible, too short)."""


def confidence_level(p: float) -> str:
    distance = abs(p - 0.5)
    return "High" if distance >= 0.3 else "Moderate" if distance >= 0.15 else "Low"


class GaitScreener:
    def __init__(self, model_dir: Path = config.MODELS):
        self.model_dir = Path(model_dir)
        self.pose_bundle = joblib.load(self.model_dir / "pose_rf.joblib")
        framing = self.model_dir / "framing_rf.joblib"
        self.framing_bundle = joblib.load(framing) if framing.exists() else None

    def analyse(self, video_path: str | Path, max_seconds: float = 45.0, progress=None) -> dict:
        try:
            extracted = pose.extract_pose(video_path, max_seconds=max_seconds, progress=progress)
        except ValueError as error:
            raise VideoError("This file could not be read as a video. Try an MP4, MOV, AVI, MKV or WEBM file.") from error
        poses, quality = pose.select_subject(extracted["poses"])
        try:
            windows = motion.video_windows(poses, float(extracted["aspect"]), quality["detection_rate"])
        except ValueError as error:
            raise VideoError(str(error)) from error
        if not windows:
            raise VideoError("No stretch of the video shows the walking person clearly enough. At least "
                             f"{config.WINDOW_SECONDS:.0f} s of continuous walking with the legs in view are needed.")

        names = self.pose_bundle["features"]
        window_p = self.pose_bundle["model"].predict_proba(motion.feature_matrix(windows, names))[:, 1]
        probability = float(window_p.mean())
        framing_p = None
        if self.framing_bundle is not None:
            framing_p = float(self.framing_bundle["model"].predict_proba(
                motion.feature_matrix(windows, self.framing_bundle["features"]))[:, 1].mean())

        warnings = []
        lower = float(np.mean([w["lower_visibility"] for w in windows]))
        if lower < 0.6:
            warnings.append("The legs are only partly visible, so the gait measurements are unreliable.")
        if quality["frames_with_other_people"] > 0.1 * extracted["poses"].shape[0]:
            warnings.append("Another person is in view for part of the video; only the person near the main position "
                            "was analysed.")
        if quality["detection_rate"] < 0.3:
            warnings.append(f"The person is in view for only {quality['detection_rate']:.0%} of the video; "
                            "only those stretches were scored.")
        if len(windows) < 3:
            warnings.append("Only a short stretch of walking could be scored, so the result is less reliable.")
        confidence = confidence_level(probability)
        shortcut = framing_p is not None and ((framing_p >= 0.5) == (probability >= 0.5)) and abs(framing_p - 0.5) >= 0.3
        if shortcut:
            warnings.append("A model that sees only camera framing (no gait) reaches the same verdict with high "
                            "confidence, so this result can be explained by how the video was filmed.")
            confidence = "Low"
        return {"probability_pd": probability, "probability_healthy": 1.0 - probability,
                "label": "Parkinson's" if probability >= 0.5 else "Healthy", "confidence": confidence,
                "framing_only_probability_pd": framing_p, "shortcut_risk": bool(shortcut),
                "window_probabilities": window_p.tolist(), "windows": len(windows), "warnings": warnings,
                "video": {"seconds_analysed": float(extracted["t"][-1]), "fps": extracted["src_fps"],
                          "width": extracted["width"], "height": extracted["height"],
                          "detection_rate": quality["detection_rate"]},
                "features": {n: float(np.nanmedian([w.get(n, np.nan) for w in windows])) for n in names},
                "poses": poses, "times": extracted["t"], "disclaimer": config.DISCLAIMER}

    @staticmethod
    def skeleton_frames(video_path: str | Path, result: dict, count: int = 6, width: int = 360) -> list[np.ndarray]:
        """A few RGB frames with the tracked skeleton drawn on them."""
        from mediapipe.tasks.python import vision

        links = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
        poses, times = result["poses"], result["times"]
        found = np.flatnonzero(np.isfinite(poses[:, 25, 0]))
        if len(found) == 0:
            return []
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        for i in found[np.linspace(0, len(found) - 1, min(count, len(found))).astype(int)]:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(times[i]) * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (width, int(frame.shape[0] * width / frame.shape[1])))
            pts = [(int(x * frame.shape[1]), int(y * frame.shape[0]), v) for x, y, _, v in poses[i]]
            for a, b in links:
                if pts[a][2] > 0.4 and pts[b][2] > 0.4:
                    cv2.line(frame, pts[a][:2], pts[b][:2], (0, 255, 0), 2)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        return frames
