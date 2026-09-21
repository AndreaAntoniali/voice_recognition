from functools import lru_cache
import os
from pathlib import Path

import torch

from manual_branch.config import DEFAULT_CHECKPOINT
from manual_branch.src.audio import adjust_duration, load_wav
from manual_branch.src.checkpoints import load_checkpoint
from manual_branch.src.log_mel import LogMelExtractor


class WavEncoder:
    """Charge un checkpoint une fois ; encode des WAV avec un crop central."""

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, device="cpu"):
        self.device = torch.device(device)
        self.model, self.config, self.metadata = load_checkpoint(checkpoint_path, self.device)
        self.extractor = LogMelExtractor(self.config).to(self.device).eval()

    @torch.inference_mode()
    def encode_wav(self, wav_path: str | Path) -> torch.Tensor:
        waveform = adjust_duration(load_wav(wav_path, self.config), config=self.config)
        log_mel = self.extractor(waveform.to(self.device))
        embedding = self.model(log_mel.unsqueeze(0))[0].cpu().float()
        if embedding.shape != (192,) or not torch.isfinite(embedding).all():
            raise ValueError("Le modèle a produit un embedding invalide.")
        if not torch.isclose(embedding.norm(), torch.tensor(1.0), atol=1e-5):
            raise ValueError("Le modèle a produit un embedding non normalisé.")
        return embedding


@lru_cache(maxsize=1)
def _default_encoder(checkpoint: str, mtime_ns: int, size: int) -> WavEncoder:
    # La signature du fichier invalide le cache après un nouvel entraînement.
    return WavEncoder(checkpoint)


def encode_wav(wav_path: str | Path) -> torch.Tensor:
    """WAV -> tenseur CPU float32 [192], L2=1, sans calcul de gradients.

    Checkpoint : MANUAL_BRANCH_CHECKPOINT ou manual_branch/models/best.pt.
    Le chargement est différé : importer le module n'exige pas de checkpoint.
    """
    checkpoint = Path(os.environ.get("MANUAL_BRANCH_CHECKPOINT", DEFAULT_CHECKPOINT)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint}. Lancez d'abord l'entraînement.")
    stat = checkpoint.stat()
    return _default_encoder(str(checkpoint), stat.st_mtime_ns, stat.st_size).encode_wav(wav_path)
