from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from manual_branch.config import AudioConfig


def load_wav(wav_path: str | Path, config: AudioConfig = AudioConfig()) -> torch.Tensor:
    """Lit un WAV mono 16 kHz sans rééchantillonnage ni conversion stéréo."""
    path = Path(wav_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Fichier WAV introuvable : {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Extension .wav requise : {path}")
    try:
        with sf.SoundFile(path) as audio:
            if audio.format not in {"WAV", "WAVEX", "RF64"}:
                raise ValueError(f"Conteneur WAV requis, reçu {audio.format} : {path}")
            if audio.channels != 1:
                raise ValueError(f"Audio mono requis, reçu {audio.channels} canaux : {path}")
            if audio.samplerate != config.sample_rate:
                raise ValueError(
                    f"Fréquence {config.sample_rate} Hz requise, reçue {audio.samplerate} Hz : {path}"
                )
            samples = audio.read(dtype="float32", always_2d=False)
    except (sf.LibsndfileError, OSError) as exc:
        raise ValueError(f"Fichier WAV invalide ou illisible : {path} ({exc})") from exc
    if samples.size == 0:
        raise ValueError(f"Fichier WAV vide : {path}")
    if not np.isfinite(samples).all():
        raise ValueError(f"Le WAV contient des NaN ou des infinis : {path}")
    return torch.from_numpy(samples.copy())


def adjust_duration(
    waveform: torch.Tensor,
    training: bool = False,
    config: AudioConfig = AudioConfig(),
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Crop aléatoire à l'entraînement, central sinon ; padding à droite."""
    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("L'audio doit être un tenseur mono non vide de forme [échantillons].")
    if not torch.isfinite(waveform).all():
        raise ValueError("L'audio contient des NaN ou des infinis.")
    waveform = waveform.float()
    extra = waveform.numel() - config.num_samples
    if extra < 0:
        return F.pad(waveform, (0, -extra))
    start = int(torch.randint(extra + 1, (), generator=generator)) if training else extra // 2
    return waveform[start : start + config.num_samples]
