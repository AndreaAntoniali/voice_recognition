"""Construit des CSV stratifiés depuis audio/<locuteur>/*.wav."""
import argparse
import csv
from pathlib import Path
import random
import re


def prepare(audio_dir, output_dir, seed=42):
    audio_dir, output_dir = Path(audio_dir), Path(output_dir)
    # Organisation acceptée : audio/<source>/<locuteur ou session locuteur>/*.wav
    # Exemples : Tel-Caroline/nabil2/*.wav -> label nabil.
    groups = {}
    for source in sorted(p for p in audio_dir.iterdir() if p.is_dir()):
        for folder in sorted(p for p in source.iterdir() if p.is_dir()):
            files = sorted(folder.glob("*.wav"))
            if not files:
                continue
            label = re.sub(r"\d+$", "", folder.name).lower()
            if not label:
                raise ValueError(f"Impossible de déduire le locuteur de {folder}")
            groups.setdefault(label, []).extend((source.name, path) for path in files)
    if not groups:
        raise ValueError(f"Aucun WAV trouvé dans des sous-dossiers de {audio_dir}.")
    speakers = {}
    for label, entries in sorted(groups.items()):
        if len(entries) < 3:
            raise ValueError(f"Le locuteur {label} doit avoir au moins 3 WAV.")
        stable_name = sum((index + 1) * ord(char) for index, char in enumerate(label))
        random.Random(seed + stable_name).shuffle(entries)
        n = len(entries)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        speakers[label] = (entries[: n - n_val - n_test], entries[n - n_val - n_test : n - n_test], entries[n - n_test :])
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, index in (("train", 0), ("validation", 1), ("test", 2)):
        with (output_dir / f"{split}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["wav_path", "label"])
            for label, groups in speakers.items():
                for _, path in groups[index]:
                    writer.writerow([str(path.relative_to(audio_dir.parent)), label])
    print("Locuteurs détectés :", {label: sum(len(group) for group in groups) for label, groups in speakers.items()})
    print("Répartition créée :", {split: sum(len(groups[index]) for groups in speakers.values()) for split, index in (("train", 0), ("validation", 1), ("test", 2))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", type=Path, default=Path("manual_branch/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("manual_branch/data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.audio_dir, args.output_dir, args.seed)
