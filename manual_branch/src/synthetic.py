"""Fixtures de diagnostic uniquement : ces cibles ne sont PAS issues d'ECAPA."""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


def create_synthetic_dataset(root: str | Path, count: int = 8, seed: int = 123) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(count):
        # Timbres et rythmes distincts ; durées courte, exacte et longue.
        duration = (2.5, 4.0, 5.0)[index % 3]
        t = np.arange(round(16000 * duration), dtype=np.float64) / 16000
        base = 100 + 85 * index
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * (index + 1) * t) ** 2
        waveform = envelope * (
            0.32 * np.sin(2 * np.pi * base * t)
            + 0.16 * np.sin(2 * np.pi * (2.3 * base + 35) * t)
            + 0.08 * np.sin(2 * np.pi * (3.7 * base + 70) * t)
        )
        wav_path = root / f"synthetic_{index:03d}.wav"
        sf.write(wav_path, waveform.astype(np.float32), 16000, subtype="FLOAT")
        rows.append((str(wav_path.resolve()), f"synthetic_{index % 2}"))
    csv_path = root / "synthetic.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wav_path", "label"])
        writer.writerows(rows)
    return csv_path
