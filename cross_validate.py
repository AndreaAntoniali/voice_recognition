"""Validation croisée « leave-one-sample-out » sur les zips 16 kHz.

`python cross_validate.py [--source audacity|python]` : `audacity` (défaut) ou `python`
(le dossier de segments découpés avec Python, ~20-30 segments par locuteur et sample).

Pour chacun des 3 samples : entraîne sur les 2 autres, teste sur celui-ci, affiche
accuracy / matrice de confusion / classification report, puis trace les 3 matrices de
confusion côte à côte (models/cv_<source>_confusion_matrices.png). Le dataset d'origine n'est pas utilisé, et rasim est exclu (il n'a un zip 16 kHz que pour le sample 1).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from preprocess import (
    AUDIO_ROOT,
    ZIP_GLOBS,
    Dataset,
    build_dataset,
    iter_zip_wavs,
    normalize,
    subset,
)
from train import MODELS_DIR, evaluate_tensors, fit

EXCLUDED = frozenset({"rasim"})
SAMPLES = ["sample1", "sample2", "sample3"]


def select(data: Dataset, samples: list[str]) -> Dataset:
    """Garde les segments dont la source (sample) figure dans `samples`."""
    return subset(data, [i for i, s in enumerate(data.source) if s in samples])


def main(source: str) -> None:
    """Lance les 3 folds leave-one-sample-out pour la source `source` ("audacity" ou
    "python"), affiche les résultats par fold puis un récapitulatif, et sauvegarde la
    figure des 3 matrices de confusion dans `models/`."""
    zip_glob = ZIP_GLOBS[source]
    figure_path = MODELS_DIR / f"cv_{source}_confusion_matrices.png"
    classes = sorted({c for c, *_ in iter_zip_wavs(AUDIO_ROOT, EXCLUDED, zip_glob)})
    label_map = {name: idx for idx, name in enumerate(classes)}
    print(f"Classes : {classes}")

    data = build_dataset(iter_zip_wavs(AUDIO_ROOT, EXCLUDED, zip_glob), label_map)

    accuracies: list[float] = []
    matrices: list[np.ndarray] = []
    for test_sample in SAMPLES:
        train_samples = [s for s in SAMPLES if s != test_sample]
        print(f"\n{'#' * 70}\n# Fold : train={train_samples} / test={test_sample}\n{'#' * 70}")

        train, test = select(data, train_samples), select(data, [test_sample])
        mean, std = train.X.mean().item(), train.X.std().item()  # stats du train uniquement
        train, test = normalize(train, mean, std), normalize(test, mean, std)

        model_path = MODELS_DIR / f"cv_{source}_{test_sample}.pt"
        fit(train.X, train.y, label_map, model_path)
        acc, cm = evaluate_tensors(
            model_path, test.X, test.y, label_map, f"Test {test_sample} (train {train_samples})"
        )
        accuracies.append(acc)
        matrices.append(cm)

    print("\n=== Récapitulatif ===")
    for sample, acc in zip(SAMPLES, accuracies):
        print(f"test {sample} : {acc:.3f}")
    print(f"moyenne : {np.mean(accuracies):.3f} (écart-type {np.std(accuracies):.3f})")

    fig, axes = plt.subplots(
        1, len(SAMPLES), figsize=(5 * len(SAMPLES), 4.6), constrained_layout=True
    )
    vmax = max(cm.max() for cm in matrices)
    for ax, sample, acc, cm in zip(axes, SAMPLES, accuracies, matrices):
        disp = ConfusionMatrixDisplay(cm, display_labels=classes)
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        disp.im_.set_clim(0, vmax)
        ax.set_title(f"Test {sample} - acc {acc:.1%}")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(f"Leave-one-sample-out ({source}) : lignes = vrai, colonnes = prédit")
    fig.savefig(figure_path, dpi=150)
    print(f"Figure sauvegardée : {figure_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=sorted(ZIP_GLOBS), default="audacity")
    main(parser.parse_args().source)
