"""Small CNN on log-Mel spectrogram windows.

The recording is cut into 1-second windows (50% overlap); the CNN scores each
window and the subject's probability is the mean over windows. The log-Mel
front end is plain PyTorch and is shared by training and inference, so both
see identical inputs. Only the CNN body is exported to ONNX.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from . import config

WINDOW = int(config.WINDOW_SECONDS * config.SAMPLE_RATE)
HOP = int(config.WINDOW_HOP_SECONDS * config.SAMPLE_RATE)


class LogMel(nn.Module):
    """Waveform batch (B, T) -> per-window standardised log-Mel (B, n_mels, frames)."""

    def __init__(self):
        super().__init__()
        import librosa

        basis = librosa.filters.mel(sr=config.SAMPLE_RATE, n_fft=config.N_FFT, n_mels=config.N_MELS,
                                    fmin=config.BANDPASS_HZ[0], fmax=config.BANDPASS_HZ[1])
        self.register_buffer("basis", torch.tensor(basis, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(config.N_FFT))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(wave, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, window=self.window,
                          center=True, return_complex=True).abs() ** 2
        mel = torch.log(torch.matmul(self.basis, spec) + 1e-6)
        mean = mel.mean(dim=(1, 2), keepdim=True)
        std = mel.std(dim=(1, 2), keepdim=True)
        return (mel - mean) / (std + 1e-5)


class MelCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()

        def block(i, o, pool=True):
            layers = [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), nn.Dropout2d(0.1))
        self.last_conv = block(64, 64, pool=False)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, n_mels, frames)
        return self.head(self.last_conv(self.features(x))).squeeze(1)


def windows(y: np.ndarray) -> np.ndarray:
    """Overlapping 1-second windows; short signals are zero-padded to one window."""
    if len(y) < WINDOW:
        y = np.pad(y, (0, WINDOW - len(y)))
    starts = range(0, len(y) - WINDOW + 1, HOP)
    return np.stack([y[s: s + WINDOW] for s in starts]).astype(np.float32)


def pitch_shifted_copies(y: np.ndarray, steps=(-1.0, -0.5, 0.5, 1.0)) -> list[np.ndarray]:
    import librosa

    return [librosa.effects.pitch_shift(y, sr=config.SAMPLE_RATE, n_steps=s).astype(np.float32) for s in steps]


def spec_augment(mel: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    mel = mel.clone()
    for i in range(mel.shape[0]):
        f = rng.integers(0, 9)
        f0 = rng.integers(0, mel.shape[1] - f + 1)
        t = rng.integers(0, 16)
        t0 = rng.integers(0, mel.shape[2] - t + 1)
        mel[i, f0:f0 + f, :] = 0.0
        mel[i, :, t0:t0 + t] = 0.0
    return mel


def add_noise(wave: torch.Tensor, rng: np.random.Generator, snr_db=(15.0, 40.0)) -> torch.Tensor:
    snr = torch.tensor(rng.uniform(*snr_db, size=(wave.shape[0], 1)), dtype=torch.float32)
    power = wave.pow(2).mean(dim=1, keepdim=True)
    noise = torch.randn_like(wave) * torch.sqrt(power / (10 ** (snr / 10)))
    return wave + noise


def build_windows(signals: dict[str, np.ndarray], subjects, labels, augment: bool):
    """Stack windows for the given subjects; returns waves, labels and subject index."""
    waves, ys, owners = [], [], []
    for index, (subject, label) in enumerate(zip(subjects, labels)):
        versions = [signals[subject]] + (pitch_shifted_copies(signals[subject]) if augment else [])
        for version in versions:
            w = windows(version)
            waves.append(w)
            ys.append(np.full(len(w), label, dtype=np.float32))
            owners.append(np.full(len(w), index))
    return np.concatenate(waves), np.concatenate(ys), np.concatenate(owners)


@torch.no_grad()
def subject_probabilities(model: MelCNN, front: LogMel, waves: np.ndarray, owners: np.ndarray,
                          n_subjects: int) -> np.ndarray:
    model.eval()
    logits = []
    for start in range(0, len(waves), 256):
        mel = front(torch.tensor(waves[start: start + 256]))
        logits.append(model(mel[:, None]))
    p = torch.sigmoid(torch.cat(logits)).numpy()
    return np.array([p[owners == i].mean() for i in range(n_subjects)])


@torch.no_grad()
def window_loss(model: MelCNN, front: LogMel, waves: np.ndarray, labels: np.ndarray) -> float:
    model.eval()
    logits = torch.cat([model(front(torch.tensor(waves[s: s + 256]))[:, None]) for s in range(0, len(waves), 256)])
    return float(nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor(labels)))


def train_model(signals, train_subjects, train_labels, val_subjects, val_labels, max_epochs: int = 100,
                patience: int = 15, min_epochs: int = 10, lr: float = 1e-3, batch_size: int = 32,
                seed: int = config.SEED, log=None) -> tuple[MelCNN, dict]:
    """Train with augmentation, class weighting, cosine LR schedule and early stopping.

    Early stopping watches the validation loss over all validation windows,
    not the subject-level AUC: with only 10-13 validation people the AUC jumps
    around from epoch to epoch and tends to pick a barely trained network.
    Weights are only eligible after ``min_epochs`` so the network has trained.
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    front = LogMel()
    model = MelCNN()
    train_w, train_y, _ = build_windows(signals, train_subjects, train_labels, augment=True)
    val_w, val_y, val_owner = build_windows(signals, val_subjects, val_labels, augment=False)
    positives = max(1.0, float(train_y.sum()))
    pos_weight = torch.tensor((len(train_y) - positives) / positives)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr / 50)

    best_loss, best_auc, best_state, best_epoch, wait, history = float("inf"), float("nan"), None, 0, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_w))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = order[start: start + batch_size]
            wave = add_noise(torch.tensor(train_w[idx]), rng)
            mel = spec_augment(front(wave), rng)
            loss = loss_fn(model(mel[:, None]), torch.tensor(train_y[idx]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()
        p_val = subject_probabilities(model, front, val_w, val_owner, len(val_subjects))
        auc = float(roc_auc_score(val_labels, p_val)) if len(set(val_labels)) == 2 else float("nan")
        val_loss = window_loss(model, front, val_w, val_y)
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_loss": val_loss, "val_auc": auc,
                        "lr": optimizer.param_groups[0]["lr"]})
        if epoch >= min_epochs:
            if val_loss < best_loss - 1e-4:
                best_loss, best_auc, best_state, best_epoch, wait = val_loss, auc, copy.deepcopy(model.state_dict()), epoch, 0
            else:
                wait += 1
        if log:
            log(f"    epoch {epoch:3d} loss {total / len(order):.4f} val loss {val_loss:.4f} val AUC {auc:.3f}")
        if wait >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "best_val_auc": best_auc, "best_epoch": best_epoch,
                   "epochs_run": len(history), "history": history}


def fit_platt(p: np.ndarray, y: np.ndarray) -> dict:
    """Logistic (Platt) calibration of subject probabilities: p' = sigmoid(a * logit(p) + b)."""
    from sklearn.linear_model import LogisticRegression

    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    model = LogisticRegression(C=10.0).fit(logit[:, None], y)  # mild L2 so few points cannot give extreme slopes
    return {"a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}


def apply_platt(p, calibration: dict | None):
    if not calibration:
        return p
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(calibration["a"] * np.log(p / (1 - p)) + calibration["b"])))


def train_fixed_epochs(signals, subjects, labels, epochs: int, lr: float = 1e-3, batch_size: int = 32,
                       seed: int = config.SEED) -> MelCNN:
    """Train on every given subject for a set number of epochs (no validation split).

    Used to refit the deployed CNN on all people once cross-validation has
    shown how many epochs early stopping typically keeps.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    front, model = LogMel(), MelCNN()
    train_w, train_y, _ = build_windows(signals, subjects, labels, augment=True)
    positives = max(1.0, float(train_y.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train_y) - positives) / positives))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 50)
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(train_w))
        for start in range(0, len(order), batch_size):
            idx = order[start: start + batch_size]
            mel = spec_augment(front(add_noise(torch.tensor(train_w[idx]), rng)), rng)
            loss = loss_fn(model(mel[:, None]), torch.tensor(train_y[idx]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()
    return model


def export_onnx(model: MelCNN, path) -> None:
    model.eval()
    dummy = torch.zeros(1, 1, config.N_MELS, WINDOW // config.HOP_LENGTH + 1)
    torch.onnx.export(model, dummy, str(path), input_names=["log_mel"], output_names=["logit"],
                      dynamic_axes={"log_mel": {0: "batch"}, "logit": {0: "batch"}}, opset_version=17,
                      dynamo=False)
