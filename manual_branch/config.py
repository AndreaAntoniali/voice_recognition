from dataclasses import dataclass
from pathlib import Path

BRANCH_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = BRANCH_ROOT / "models" / "best.pt"
EMBEDDING_DIM = 192


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    duration_seconds: float = 4.0
    n_fft: int = 400
    win_length: int = 400
    hop_length: int = 160
    n_mels: int = 80
    f_min: float = 50.0
    f_max: float = 7600.0
    top_db: float = 80.0
    eps: float = 1e-6

    def __post_init__(self):
        # Le contrat de cette branche est fixe, y compris au rechargement.
        expected = (16000, 4.0, 400, 400, 160, 80, 50.0, 7600.0, 80.0, 1e-6)
        actual = tuple(getattr(self, key) for key in self.__dataclass_fields__)
        if actual != expected:
            raise ValueError("Configuration audio incompatible avec le contrat de cette branche.")

    @property
    def num_samples(self) -> int:
        return round(self.sample_rate * self.duration_seconds)
