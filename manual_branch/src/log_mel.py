import torch
from torch import nn

from manual_branch.config import AudioConfig


class LogMelExtractor(nn.Module):
    """Spectre de puissance, filtres HTK triangulaires, dB puis z-score global.

    La normalisation porte sur les axes Mel et temps de chaque fichier,
    indépendamment du batch. Le silence donne un spectrogramme nul et fini.
    """

    def __init__(self, config: AudioConfig = AudioConfig()):
        super().__init__()
        self.config = config
        self.register_buffer("window", torch.hann_window(config.win_length))
        limits = 2595.0 * torch.log10(1.0 + torch.tensor([config.f_min, config.f_max]) / 700.0)
        mel_points = torch.linspace(limits[0], limits[1], config.n_mels + 2)
        hz = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        frequencies = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
        rising = (frequencies[None, :] - hz[:-2, None]) / (hz[1:-1] - hz[:-2])[:, None]
        falling = (hz[2:, None] - frequencies[None, :]) / (hz[2:] - hz[1:-1])[:, None]
        self.register_buffer("mel_filters", torch.minimum(rising, falling).clamp_min(0))

    def to_db(self, waveform: torch.Tensor) -> torch.Tensor:
        """[64000] ou [batch, 64000] -> [80, 401] ou [batch, 80, 401]."""
        if waveform.ndim not in (1, 2) or waveform.shape[-1] != self.config.num_samples:
            raise ValueError("Le Log-Mel attend [64000] ou [batch, 64000] après ajustement.")
        if not torch.isfinite(waveform).all():
            raise ValueError("L'audio contient des NaN ou des infinis.")
        spectrum = torch.stft(
            waveform.float(), n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, win_length=self.config.win_length,
            window=self.window, center=True, pad_mode="reflect", return_complex=True,
        )
        power = spectrum.abs().square()
        mel = self.mel_filters @ power
        db = 10.0 * torch.log10(mel.clamp_min(1e-10))
        return torch.maximum(db, db.amax(dim=(-2, -1), keepdim=True) - self.config.top_db)

    def normalize(self, db: torch.Tensor) -> torch.Tensor:
        mean = db.mean(dim=(-2, -1), keepdim=True)
        std = db.std(dim=(-2, -1), keepdim=True, unbiased=False)
        return (db - mean) / std.clamp_min(self.config.eps)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.normalize(self.to_db(waveform))
