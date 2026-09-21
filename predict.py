"""Prédit le locuteur d'un enregistrement audio avec un modèle entraîné.

`python predict.py enregistrement.m4a [--model models/best_model_python.pt]`

Le modèle par défaut est `models/best_model_python.pt`, créé par
`python train.py --source python`.

Pipeline : audio -> mono 16 kHz -> troncature des silences -> segments de 3 s normalisés
LUFS (`audio_processing.py`, comme pour les données d'entraînement) -> spectrogramme de
Mel (dB) -> normalisation avec les statistiques du train -> CNN. Chaque segment reçoit
une prédiction ; le locuteur retenu est celui dont la probabilité moyenne est la plus
élevée sur l'ensemble des segments.
"""

import argparse
from pathlib import Path

import torch

from audio_processing import charger_audio, decouper_en_segments
from model import SpeakerCNN
from preprocess import (
    OUTPUT_DIR,
    NORM_EPS,
    TARGET_LENGTH,
    fix_length,
    make_mel_transforms,
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL = MODELS_DIR / "best_model_python.pt"


def load_model(model_path: Path) -> tuple[SpeakerCNN, dict[str, int], float, float]:
    """Charge un checkpoint produit par `train.fit`.

    Returns:
        `(modèle en mode eval, label_map, mean, std)`. Les anciens checkpoints sans
        statistiques de normalisation sont acceptés en reprenant celles de
        `data/train_data.pt` (valable pour `best_model.pt`, qui est entraîné dessus).
    """
    checkpoint = torch.load(model_path)
    label_map = checkpoint["label_map"]

    if "mean" in checkpoint:
        mean, std = checkpoint["mean"], checkpoint["std"]
    else:
        stats = torch.load(OUTPUT_DIR / "train_data.pt")
        mean, std = stats["mean"], stats["std"]
        print(
            f"[attention] {model_path.name} ne contient pas ses statistiques de normalisation : "
            "utilisation de celles de data/train_data.pt (à vérifier pour les checkpoints "
            "de validation croisée, ré-entraînez-les pour les stocker)."
        )

    model = SpeakerCNN(num_classes=len(label_map))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, label_map, mean, std


def audio_to_spectrograms(audio_path: Path, mean: float, std: float) -> torch.Tensor:
    """Transforme un fichier audio en batch de spectrogrammes normalisés.

    Returns:
        Tensor `(N, 1, n_mels, T)`, un par segment de 3 s.

    Raises:
        ValueError: si l'audio ne contient aucun segment complet de 3 s (après troncature
            des silences).
    """
    segments = decouper_en_segments(charger_audio(str(audio_path)))
    if not segments:
        raise ValueError(
            f"{audio_path.name} : moins de 3 s d'audio exploitable après troncature des "
            "silences, aucun segment à classer."
        )

    mel_transform, db_transform = make_mel_transforms()
    specs = []
    for samples in segments:
        waveform = torch.from_numpy(samples).float().unsqueeze(0) / 32768.0  # (1, 48000)
        waveform = fix_length(waveform, TARGET_LENGTH)
        specs.append(db_transform(mel_transform(waveform)))
    X = torch.stack(specs)
    return (X - mean) / (std + NORM_EPS)


def predict(audio_path: Path, model_path: Path = DEFAULT_MODEL) -> None:
    """Classe chaque segment de `audio_path` et affiche le détail puis le locuteur retenu."""
    if not model_path.exists():
        raise SystemExit(
            f"Modèle introuvable : {model_path}\n"
            "Entraînez-le avec `python train.py --source python` ou choisissez un autre "
            "checkpoint avec --model."
        )
    model, label_map, mean, std = load_model(model_path)
    idx_to_name = {idx: name for name, idx in label_map.items()}

    X = audio_to_spectrograms(audio_path, mean, std)
    with torch.no_grad():
        probs = torch.softmax(model(X), dim=1)  # (N, num_classes)

    print(f"\n{audio_path.name} : {len(X)} segment(s) de 3 s, modèle {model_path.name}\n")
    for i, p in enumerate(probs, start=1):
        best = p.argmax().item()
        print(f"  segment {i:02d} : {idx_to_name[best]:<10} ({p[best]:.1%})")

    mean_probs = probs.mean(dim=0)
    print("\nProbabilité moyenne par locuteur :")
    for idx in mean_probs.argsort(descending=True).tolist():
        print(f"  {idx_to_name[idx]:<10} {mean_probs[idx]:.1%}")
    print(f"\n=> Locuteur prédit : {idx_to_name[mean_probs.argmax().item()]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, help="fichier audio (.m4a, .wav, ...)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="checkpoint .pt")
    args = parser.parse_args()
    predict(args.audio, args.model)
