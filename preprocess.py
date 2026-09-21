"""Prétraitement des segments audio en spectrogrammes de Mel pour l'entraînement du CNN.

Pipeline : WAV -> resampling 16kHz -> longueur fixe -> spectrogramme de Mel (dB)
-> split train/test stratifié -> normalisation (mean 0 / std 1) -> sauvegarde .pt.

Deux sources de données :
- le dataset d'origine (`audio_files/audio_files/<nom>_processed/<nom>/*.wav`), splitté
  en train/test comme avant (le test set reste donc identique) ;
- les enregistrements supplémentaires en 16 kHz, lus directement dans les zips
  (`.../Audacity/16000Hz/sample<k>_*/<nom>-<k>.zip`) sans extraction : les samples
  d'entraînement (1, 2) sont ajoutés au train, le sample HELD_OUT_SAMPLE est un test
  set séparé (autre session d'enregistrement, donc pas de fuite entre segments voisins).
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
ZIP_GLOB = ZIP_GLOBS["audacity"]
EXCLUDE_CLASSES: frozenset[str] = frozenset()  # classes à ignorer (ex. {"rasim"})
# Classes exclues pour les entraînements sur les zips seuls (validation croisée, modèle
# final) : rasim n'a un zip que pour le sample 1.
FINAL_EXCLUDED_CLASSES: frozenset[str] = frozenset({"rasim"})
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
# ou le nom du sample ("sample1", ...) pour les zips.
AudioItem = tuple[str, str, str, "io.IOBase"]


def iter_original_wavs(dataset_root: Path) -> Iterator[AudioItem]:
    """Dataset d'origine, pattern `<nom>_processed/<nom>/*.wav`."""
    for class_dir in sorted(dataset_root.glob("*_processed")):
        class_name = class_dir.name.removesuffix("_processed")
        for wav_path in sorted(class_dir.glob("*/*.wav")):
            with open(wav_path, "rb") as f:
                yield class_name, "original", wav_path.name, f


def iter_zip_wavs(
    audio_root: Path,
    exclude: frozenset[str] = EXCLUDE_CLASSES,
    zip_glob: str = ZIP_GLOB,
) -> Iterator[AudioItem]:
    """WAV 16 kHz lus directement dans les zips, sans extraction sur disque.
    La classe se déduit du nom du zip (`Andrea-1.zip` / `andrea-1.zip` -> `andrea`)."""
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
    """Lit un WAV PCM 16-bit (chemin ou fichier ouvert) via le module standard `wave`
    (torchaudio.load est cassé dans cet environnement, cf. plan) et retourne
    (waveform (1, N), sample_rate)."""
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
    """Cache un `Resample` par fréquence source rencontrée."""
    if orig_sr not in _resampler_cache:
        _resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, TARGET_SAMPLE_RATE)
    return _resampler_cache[orig_sr]


def fix_length(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad avec des zéros si trop court, tronque si trop long."""
    current_length = waveform.shape[-1]
    if current_length < target_length:
        return F.pad(waveform, (0, target_length - current_length))
    return waveform[..., :target_length]


def make_mel_transforms() -> tuple[
    torchaudio.transforms.MelSpectrogram, torchaudio.transforms.AmplitudeToDB
]:
    """Construit les transformations Mel + dB avec les paramètres du projet. Partagé par
    le prétraitement et l'inférence (`predict.py`) pour garantir les mêmes features."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    return mel_transform, db_transform


def process_file(
    wav_file: "io.IOBase",
    name: str,
    mel_transform: torchaudio.transforms.MelSpectrogram,
    db_transform: torchaudio.transforms.AmplitudeToDB,
) -> torch.Tensor | None:
    """Charge et transforme un fichier en spectrogramme de Mel (dB). Retourne None
    si le fichier est un reste de découpage (durée trop éloignée de 3.0s)."""
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
    X: torch.Tensor  # (N, 1, N_MELS, T)
    y: torch.Tensor  # (N,)
    source: list[str]  # source de chaque échantillon ("original", "sample1", ...)


def build_dataset(items: Iterator[AudioItem], label_map: dict[str, int]) -> Dataset:
    mel_transform, db_transform = make_mel_transforms()

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


def build_zip_dataset(
    source: str, exclude: frozenset[str] = FINAL_EXCLUDED_CLASSES
) -> tuple[Dataset, dict[str, int]]:
    """Construit le `Dataset` (non normalisé) de tous les samples d'une source de zips.

    Args:
        source: clé de `ZIP_GLOBS` ("audacity" ou "python").
        exclude: classes à ignorer.

    Returns:
        `(dataset, label_map)`, le `label_map` étant construit sur les classes trouvées
        (triées par ordre alphabétique).
    """
    zip_glob = ZIP_GLOBS[source]
    classes = sorted({c for c, *_ in iter_zip_wavs(AUDIO_ROOT, exclude, zip_glob)})
    if not classes:
        raise RuntimeError(f"Aucun zip trouvé pour la source '{source}' sous {AUDIO_ROOT}")
    label_map = {name: idx for idx, name in enumerate(classes)}
    print(f"Classes : {classes}")
    return build_dataset(iter_zip_wavs(AUDIO_ROOT, exclude, zip_glob), label_map), label_map


# --- Split et normalisation ---------------------------------------------------


def stratified_split(data: Dataset) -> tuple[Dataset, Dataset]:
    indices = list(range(len(data.y)))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=data.y.tolist()
    )
    return subset(data, train_idx), subset(data, test_idx)


def subset(data: Dataset, indices: list[int]) -> Dataset:
    idx = torch.tensor(indices, dtype=torch.long)
    return Dataset(data.X[idx], data.y[idx], [data.source[i] for i in indices])


def normalize(data: Dataset, mean: float, std: float) -> Dataset:
    return Dataset((data.X - mean) / (std + NORM_EPS), data.y, data.source)


# --- Sauvegarde ---------------------------------------------------------------


def save_split(
    path: Path,
    data: Dataset,
    label_map: dict[str, int],
    mean: float,
    std: float,
) -> None:
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
    counts = torch.bincount(y, minlength=len(label_map))
    idx_to_name = {idx: n for n, idx in label_map.items()}
    distribution = ", ".join(f"{idx_to_name[i]}={counts[i].item()}" for i in range(len(counts)))
    print(f"Répartition {name}: {distribution}")


def main() -> None:
    classes = {c for c, *_ in iter_original_wavs(DATASET_ROOT)}
    classes |= {c for c, *_ in iter_zip_wavs(AUDIO_ROOT)}
    if not classes:
        raise RuntimeError(f"Aucun fichier audio trouvé sous {AUDIO_ROOT}")
    label_map = {name: idx for idx, name in enumerate(sorted(classes))}

    print("Dataset d'origine :")
    original = build_dataset(iter_original_wavs(DATASET_ROOT), label_map)
    print("Zips 16 kHz :")
    extra = build_dataset(iter_zip_wavs(AUDIO_ROOT), label_map)

    # Le test d'origine reste identique à celui d'avant (même split, même seed) ; les
    # enregistrements supplémentaires vont soit dans le train, soit dans un test séparé.
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
