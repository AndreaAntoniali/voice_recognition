"""Dataset PyTorch pour les spectrogrammes de Mel prétraités (voir preprocess.py)."""

import random

import torch
from torch.utils.data import Dataset


def spec_augment(
    spec: torch.Tensor,
    freq_mask_param: int = 8,
    time_mask_param: int = 30,
    p: float = 0.8,
) -> torch.Tensor:
    """Masque aléatoirement une bande de fréquence et une bande temporelle (mise à
    0) avec probabilité `p`. Inspiré de SpecAugment ; utile pour régulariser
    l'entraînement sur un dataset aussi réduit. `spec` a la forme (1, n_mels, T)."""
    spec = spec.clone()

    if random.random() < p:
        n_mels = spec.shape[-2]
        f = random.randint(0, min(freq_mask_param, n_mels))
        f0 = random.randint(0, n_mels - f)
        spec[..., f0 : f0 + f, :] = 0.0

    if random.random() < p:
        n_frames = spec.shape[-1]
        t = random.randint(0, min(time_mask_param, n_frames))
        t0 = random.randint(0, n_frames - t)
        spec[..., :, t0 : t0 + t] = 0.0

    return spec


class SpeakerDataset(Dataset):
    """Wrap un tensor de spectrogrammes (N, 1, n_mels, T) et leurs labels (N,)."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor, augment: bool = False):
        self.X = X
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.augment:
            x = spec_augment(x)
        return x, self.y[idx]
