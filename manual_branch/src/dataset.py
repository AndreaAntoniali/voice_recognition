import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from manual_branch.config import AudioConfig, BRANCH_ROOT
from manual_branch.src.audio import adjust_duration, load_wav
from manual_branch.src.log_mel import LogMelExtractor


class SpeakerDataset(Dataset):
    def __init__(
        self, csv_path: str | Path, training: bool = False,
        root: str | Path = BRANCH_ROOT, config: AudioConfig = AudioConfig(),
    ):
        self.config = config
        self.training = training
        self.extractor = LogMelExtractor(config)
        root = Path(root).expanduser().resolve()
        self.records = []
        with Path(csv_path).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["wav_path", "label"]:
                raise ValueError(f"En-tête CSV attendu : wav_path,label ({csv_path})")
            for line, row in enumerate(reader, start=2):
                if None in row or any(not (row.get(key) or "").strip() for key in reader.fieldnames):
                    raise ValueError(f"Ligne CSV incomplète ou invalide : {csv_path}:{line}")
                record = {key: value.strip() for key, value in row.items()}
                path = Path(record["wav_path"]).expanduser()
                record["wav_path"] = str((path if path.is_absolute() else root / path).resolve())
                if not Path(record["wav_path"]).is_file():
                    raise FileNotFoundError(f"wav_path introuvable à la ligne {line} : {record['wav_path']}")
                self.records.append(record)
        if not self.records:
            raise ValueError(f"CSV vide : renseignez des couples WAV/ECAPA dans {csv_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        waveform = adjust_duration(load_wav(record["wav_path"], self.config), self.training, self.config)
        return {
            "log_mel": self.extractor(waveform),
            "label": record["label"],  # Identifiant opaque, aucun classifieur.
            "wav_path": record["wav_path"],
        }
