from dataclasses import asdict
import os
from pathlib import Path
import tempfile

import torch

from manual_branch.config import AudioConfig
from manual_branch.src.manual_encoder import ManualEncoder

CHECKPOINT_VERSION = 1


def save_checkpoint(path, model, config, epoch, validation_loss, optimizer=None):
    """Sauvegarde atomique : poids, prétraitement et contexte d'entraînement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "architecture": "manual_cnn_16_32_128_groupnorm_256_192",
        "audio_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "validation_loss": float(validation_loss),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path, device="cpu"):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {path}. Entraînez d'abord la branche manuelle.")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("version") != CHECKPOINT_VERSION:
            raise ValueError("Version de checkpoint non reconnue.")
        if checkpoint.get("architecture") != "manual_cnn_16_32_128_groupnorm_256_192":
            raise ValueError("Architecture de checkpoint incompatible.")
        config_dict = checkpoint["audio_config"]
        if set(config_dict) != set(AudioConfig.__dataclass_fields__):
            raise ValueError("Configuration audio absente ou incomplète.")
        config = AudioConfig(**config_dict)
        model = ManualEncoder()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
            raise ValueError("Les poids contiennent des NaN ou des infinis.")
    except Exception as exc:
        raise ValueError(f"Checkpoint invalide ou incompatible : {path} ({exc})") from exc
    model.to(device).eval().requires_grad_(False)
    return model, config, checkpoint
