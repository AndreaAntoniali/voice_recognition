"""Prétraitement des segments audio en spectrogrammes de Mel pour l'entraînement du CNN.

Pipeline : WAV -> resampling 16kHz -> longueur fixe -> spectrogramme de Mel (dB)
-> split train/test stratifié -> normalisation (mean 0 / std 1) -> sauvegarde .pt.

Deux sources de données :
- le dataset d'origine (`audio_files/audio_files/<nom>_processed/<nom>/*.wav`), splitté
  en train/test stratifié (seed fixe, donc reproductible) ;
- les enregistrements supplémentaires en 16 kHz, lus directement dans les zips sans
  extraction (`.../Audacity/16000Hz/sample<k>_*/<nom>-<k>.zip` ou
  `.../Python/sample<k>_*/<Nom>-<k>.zip`) : les samples d'entraînement (1, 2) sont
  ajoutés au train, le sample HELD_OUT_SAMPLE forme un test set séparé (autre session
  d'enregistrement, donc pas de fuite entre segments voisins).

Lancé en script (`python preprocess.py`), il écrit `data/train_data.pt`,
`data/test_data.pt` et `data/test_extra_data.pt`. Les fonctions sont aussi importées par
`cross_validate.py`.
"""

import io
import wave
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
import torchaudio
from sklearn.model_selection import train_test_split

# --- Configuration ---------------------------------------------------------

AUDIO_ROOT = Path(__file__).resolve().parent / "audio_files"
DATASET_ROOT = AUDIO_ROOT / "audio_files"
ZIP_GLOBS = {
    "audacity": "**/Audacity/16000Hz/sample*/*.zip",
    "python": "**/Python/sample*/*.zip",
}
EXCLUDE_CLASSES: frozenset[str] = frozenset()  # classes à ignorer (ex. {"rasim"})
HELD_OUT_SAMPLE = "sample3"  # sample réservé au test ; les autres vont dans le train
OUTPUT_DIR = Path(__file__).resolve().parent / "data"

TARGET_SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 3.0
TARGET_LENGTH = int(TARGET_SAMPLE_RATE * TARGET_DURATION_SEC)  # 48000 échantillons

# Le dataset est presque entièrement composé de segments de exactement 3.000s, mais
# contient un reste de découpage par locuteur (durée variable, jusqu'à 2.95s) : comme
# les segments propres sont bit-exacts, une tolérance serrée suffit à les exclure tous.
DURATION_TOLERANCE_SEC = 0.01

N_MELS = 64  # alternative standard : 128
N_FFT = 400  # fenêtre de 25 ms à 16kHz, valeur usuelle pour la parole
HOP_LENGTH = 160  # hop de 10 ms à 16kHz

TEST_SIZE = 0.2
RANDOM_SEED = 42
NORM_EPS = 1e-6


# --- Chargement des fichiers -------------------------------------------------


# Un échantillon audio : (classe, source, nom, fichier ouvert). `source` vaut "original"
# pour le dataset d'origine, ou le nom du sample ("sample1", ...) pour les zips.
AudioItem = tuple[str, str, str, "io.IOBase"]


def iter_original_wavs(dataset_root: Path) -> Iterator[AudioItem]:
    """Parcourt le dataset d'origine (pattern `<nom>_processed/<nom>/*.wav`).

    Args:
        dataset_root: dossier contenant les `<nom>_processed/`.

    Yields:
        `(classe, "original", nom_du_fichier, fichier_ouvert)`. Le fichier n'est valide
        que le temps de l'itération : le consommateur doit le lire avant de passer au
        suivant.
    """
    for class_dir in sorted(dataset_root.glob("*_processed")):
        class_name = class_dir.name.removesuffix("_processed")
        for wav_path in sorted(class_dir.glob("*/*.wav")):
            with open(wav_path, "rb") as f:
                yield class_name, "original", wav_path.name, f


def iter_zip_wavs(
    audio_root: Path,
    exclude: frozenset[str] = EXCLUDE_CLASSES,
    zip_glob: str = ZIP_GLOBS["audacity"],
) -> Iterator[AudioItem]:
    """Parcourt les WAV contenus dans les zips, sans extraction sur disque.

    La classe se déduit du nom du zip (`Andrea-1.zip` ou `andrea-1.zip` -> `andrea`) et
    la source du dossier parent (`sample1_tel-caroline` -> `sample1`). Les entrées
    `__MACOSX` et les fichiers non `.wav` sont ignorés.

    Args:
        audio_root: dossier racine dans lequel chercher les zips.
        exclude: classes à ignorer (leur zip n'est même pas ouvert).
        zip_glob: motif glob relatif à `audio_root` (voir `ZIP_GLOBS`).

    Yields:
        `(classe, sample, nom_dans_le_zip, BytesIO_du_wav)`.
    """
    for zip_path in sorted(audio_root.glob(zip_glob)):
        class_name = zip_path.stem.rsplit("-", 1)[0].lower()
        if class_name in exclude:
            continue
        source = zip_path.parent.name.split("_")[0]  # "sample1_tel-caroline" -> "sample1"
        with zipfile.ZipFile(zip_path) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".wav") and not name.startswith("__MACOSX"):
                    yield class_name, source, name, io.BytesIO(zf.read(name))


def load_wav_as_tensor(source: "str | io.IOBase") -> tuple[torch.Tensor, int]:
    """Lit un WAV PCM 16-bit via le module standard `wave`.

    `torchaudio.load` est cassé dans cet environnement, d'où ce lecteur maison. Un WAV
    multicanal est moyenné en mono.

    Args:
        source: chemin ou fichier ouvert (ex. `io.BytesIO` d'un membre de zip).

    Returns:
        `(waveform, sample_rate)` avec `waveform` de forme `(1, N)`, valeurs dans [-1, 1].

    Raises:
        ValueError: si le WAV n'est pas en 16 bits.
    """
    with wave.open(source, "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        raw_bytes = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"{source}: seul le PCM 16-bit est supporté (sampwidth={sample_width})")

    samples = torch.frombuffer(bytearray(raw_bytes), dtype=torch.int16).float() / 32768.0

    if n_channels > 1:
        samples = samples.view(-1, n_channels).mean(dim=1)

    return samples.unsqueeze(0), sample_rate  # (1, N)


# --- Transformations audio ---------------------------------------------------

_resampler_cache: dict[int, torchaudio.transforms.Resample] = {}


def get_resampler(orig_sr: int) -> torchaudio.transforms.Resample:
    """Retourne un `Resample` orig_sr -> 16 kHz, mis en cache par fréquence source."""
    if orig_sr not in _resampler_cache:
        _resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, TARGET_SAMPLE_RATE)
    return _resampler_cache[orig_sr]


def fix_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Ramène le dernier axe à `target_length` : padding de zéros à droite si trop court,
    troncature si trop long."""
    current_length = waveform.shape[-1]
    if current_length < target_length:
        return F.pad(waveform, (0, target_length - current_length))
    return waveform[..., :target_length]


def process_file(
    wav_file: "io.IOBase",
    name: str,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB,
) -> torch.Tensor | None:
    """Transforme un WAV en spectrogramme de Mel (dB).

    Args:
        wav_file: fichier WAV ouvert (voir `load_wav_as_tensor`).
        name: nom du fichier, utilisé seulement pour le message d'ignorance.
        mel_transform: transformation Mel partagée entre les fichiers.
        db_transform: conversion amplitude -> dB.

    Returns:
        Tensor `(1, N_MELS, T)`, ou `None` si la durée s'écarte de 3.0 s de plus de
        `DURATION_TOLERANCE_SEC` (reste de découpage, ignoré).
    """
    waveform, orig_sr = load_wav_as_tensor(wav_file)

    duration_sec = waveform.shape[-1] / orig_sr
    if abs(duration_sec - TARGET_DURATION_SEC) > DURATION_TOLERANCE_SEC:
        print(f"  [ignoré] {name} : durée {duration_sec:.3f}s hors tolérance")
        return None

    waveform = get_resampler(orig_sr)(waveform)
    waveform = fix_length(waveform, TARGET_LENGTH)

    mel_spec = mel_transform(waveform)
    mel_spec_db = db_transform(mel_spec)
    return mel_spec_db  # (1, N_MELS, T)


# --- Construction du dataset --------------------------------------------------


class Dataset(NamedTuple):
    """Spectrogrammes, labels et provenance de chaque segment (tous alignés sur N)."""

    X: torch.Tensor  # (N, 1, N_MELS, T)
    y: torch.Tensor  # (N,)
    source: list[str]  # source de chaque échantillon ("original", "sample1", ...)


def build_dataset(items: Iterator[AudioItem], label_map: dict[str, int]) -> Dataset:
    """Transforme tous les échantillons audio en un `Dataset` de spectrogrammes de Mel.

    Les segments hors durée sont écartés ; un résumé « gardés/vus » par (source, classe)
    est affiché.

    Args:
        items: échantillons issus de `iter_original_wavs` ou `iter_zip_wavs`.
        label_map: nom de classe -> indice entier.

    Returns:
        `Dataset` non normalisé (spectrogrammes en dB bruts).
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    specs: list[torch.Tensor] = []
    labels: list[int] = []
    sources: list[str] = []
    kept: Counter[tuple[str, str]] = Counter()
    seen: Counter[tuple[str, str]] = Counter()

    for class_name, source, name, wav_file in items:
        seen[(source, class_name)] += 1
        spec = process_file(wav_file, name, mel_transform, db_transform)
        if spec is not None:
            specs.append(spec)
            labels.append(label_map[class_name])
            sources.append(source)
            kept[(source, class_name)] += 1

    for key in sorted(seen):
        print(f"  {key[0]}/{key[1]}: {kept[key]}/{seen[key]} fichiers gardés")

    X = torch.stack(specs, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    print(f"Dataset : X={tuple(X.shape)}, y={tuple(y.shape)}")
    return Dataset(X, y, sources)


# --- Split et normalisation ---------------------------------------------------


def stratified_split(data: Dataset) -> tuple[Dataset, Dataset]:
    """Split train/test stratifié par classe (`TEST_SIZE`, seed `RANDOM_SEED`)."""
    indices = list(range(len(data.y)))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data.y.tolist()
    )
    return subset(data, train_idx), subset(data, test_idx)


def subset(data: Dataset, indices: list[int]) -> Dataset:
    """Sélectionne les segments `indices` (X, y et source restent alignés)."""
    idx = torch.tensor(indices, dtype=torch.long)
    return Dataset(data.X[idx], data.y[idx], [data.source[i] for i in indices])


def normalize(data: Dataset, mean: float, std: float) -> Dataset:
    """Normalise X avec des statistiques fournies (calculées sur le train uniquement,
    pour éviter toute fuite du test vers le train)."""
    return Dataset((data.X - mean) / (std + NORM_EPS), data.y, data.source)


# --- Sauvegarde ---------------------------------------------------------------


def save_split(
    path: Path,
    data: Dataset,
    label_map: dict[str, int],
    mean: float,
    std: float,
) -> None:
    """Sauvegarde un split en `.pt` avec ses métadonnées (label_map, statistiques de
    normalisation, paramètres du spectrogramme) pour rester autonome."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "X": data.X,
            "y": data.y,
            "source": data.source,
            "label_map": label_map,
            "mean": mean,
            "std": std,
            "sample_rate": TARGET_SAMPLE_RATE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "target_length": TARGET_LENGTH,
        },
        path,
    )
    print(f"Sauvegardé : {path} (X={tuple(data.X.shape)}, y={tuple(data.y.shape)})")


def print_class_distribution(name: str, y: torch.Tensor, label_map: dict[str, int]) -> None:
    """Affiche le nombre de segments par classe pour le split `name`."""
    counts = torch.bincount(y, minlength=len(label_map))
    idx_to_name = {idx: n for n, idx in label_map.items()}
    distribution = ", ".join(f"{idx_to_name[i]}={counts[i].item()}" for i in range(len(counts)))
    print(f"Répartition {name}: {distribution}")


def main() -> None:
    """Construit et sauvegarde les trois splits : train (original + samples d'entraînement),
    test d'origine, et test sur le sample réservé `HELD_OUT_SAMPLE`."""
    classes = {c for c, *_ in iter_original_wavs(DATASET_ROOT)}
    classes |= {c for c, *_ in iter_zip_wavs(AUDIO_ROOT)}
    if not classes:
        raise RuntimeError(f"Aucun fichier audio trouvé sous {AUDIO_ROOT}")
    label_map = {name: idx for idx, name in enumerate(sorted(classes))}

    print("Dataset d'origine :")
    original = build_dataset(iter_original_wavs(DATASET_ROOT), label_map)
    print("Zips 16 kHz :")
    extra = build_dataset(iter_zip_wavs(AUDIO_ROOT), label_map)

    # Les enregistrements supplémentaires vont soit dans le train, soit dans un test séparé.
    train, test = stratified_split(original)
    held_out_idx = [i for i, s in enumerate(extra.source) if s == HELD_OUT_SAMPLE]
    extra_train_idx = [i for i, s in enumerate(extra.source) if s != HELD_OUT_SAMPLE]
    train = Dataset(
        torch.cat([train.X, subset(extra, extra_train_idx).X]),
        torch.cat([train.y, subset(extra, extra_train_idx).y]),
        train.source + subset(extra, extra_train_idx).source,
    )
    test_extra = subset(extra, held_out_idx)

    print_class_distribution("train", train.y, label_map)
    print_class_distribution("test (original)", test.y, label_map)
    print_class_distribution(f"test ({HELD_OUT_SAMPLE})", test_extra.y, label_map)

    mean, std = train.X.mean().item(), train.X.std().item()
    save_split(OUTPUT_DIR / "train_data.pt", normalize(train, mean, std), label_map, mean, std)
    save_split(OUTPUT_DIR / "test_data.pt", normalize(test, mean, std), label_map, mean, std)
    save_split(
        OUTPUT_DIR / "test_extra_data.pt", normalize(test_extra, mean, std), label_map, mean, std
    )


if __name__ == "__main__":
    main()
