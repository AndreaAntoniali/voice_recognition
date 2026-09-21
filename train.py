"""Entraîne le CNN de reconnaissance de locuteur et l'évalue sur les test sets.

`python train.py`            : entraîne sur le dataset d'origine + les zips 16 kHz.
`python train.py --no-extra` : baseline, entraîne sur le dataset d'origine seul.
Dans les deux cas l'évaluation porte sur le même test d'origine et sur le sample
réservé (test_extra), ce qui permet de mesurer l'apport des données supplémentaires.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from dataset import SpeakerDataset
from model import SpeakerCNN

# --- Configuration -----------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"

VAL_SIZE = 0.15
RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_EPOCHS = 50
PATIENCE = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


def make_train_val_loaders(
    X: torch.Tensor, y: torch.Tensor
) -> tuple[DataLoader, DataLoader]:
    """Split train/val stratifié. Les segments d'un même enregistrement sont voisins
    dans le temps : la val_acc est donc optimiste, les test sets font foi."""
    indices = list(range(len(y)))
    train_idx, val_idx = train_test_split(
        indices, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y.tolist()
    )
    train_idx = torch.tensor(train_idx, dtype=torch.long)
    val_idx = torch.tensor(val_idx, dtype=torch.long)

    train_ds = SpeakerDataset(X[train_idx], y[train_idx], augment=True)
    val_ds = SpeakerDataset(X[val_idx], y[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Une passe d'entraînement (si optimizer fourni) ou de validation."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y_batch.size(0)
            total_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)

    return total_loss / total_samples, total_correct / total_samples


def fit(
    X: torch.Tensor, y: torch.Tensor, label_map: dict[str, int], best_model_path: Path
) -> None:
    """Entraîne un SpeakerCNN sur (X, y) avec early stopping sur un split de validation
    interne, et sauvegarde le meilleur checkpoint dans `best_model_path`."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = make_train_val_loaders(X, y)

    model = SpeakerCNN(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_model_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} - "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} - "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "label_map": label_map}, best_model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping à l'epoch {epoch} (patience={PATIENCE} atteinte)")
                break

    print(f"Meilleure val_acc: {best_val_acc:.3f} -> sauvegardé dans {best_model_path}")


def train(use_extra: bool, best_model_path: Path) -> dict[str, int]:
    train_data = torch.load(DATA_DIR / "train_data.pt")
    X_full, y_full, label_map = train_data["X"], train_data["y"], train_data["label_map"]
    if not use_extra:
        keep = torch.tensor([src == "original" for src in train_data["source"]])
        X_full, y_full = X_full[keep], y_full[keep]
    print(f"Entraînement sur {len(y_full)} segments (use_extra={use_extra})")

    fit(X_full, y_full, label_map, best_model_path)
    return label_map


def evaluate_tensors(
    best_model_path: Path,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    label_map: dict[str, int],
    title: str,
) -> tuple[float, np.ndarray]:
    """Évalue le checkpoint sur (X_test, y_test), affiche accuracy, matrice de confusion
    et classification report, et retourne (accuracy, matrice de confusion)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(best_model_path)
    model = SpeakerCNN(num_classes=len(checkpoint["label_map"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    X_test = X_test.to(device)
    with torch.no_grad():
        preds = torch.cat(
            [model(X_test[i : i + 64]).argmax(dim=1).cpu() for i in range(0, len(X_test), 64)]
        )

    idx_to_name = {idx: name for name, idx in label_map.items()}
    class_names = [idx_to_name[i] for i in range(len(idx_to_name))]

    accuracy = (preds == y_test).float().mean().item()
    print(f"\n=== {title} ({len(y_test)} segments) ===")
    print(f"Test accuracy: {accuracy:.3f}")

    print("\nMatrice de confusion (lignes=vrai, colonnes=prédit):")
    cm = confusion_matrix(y_test, preds, labels=range(len(class_names)))
    header = "        " + " ".join(f"{name[:8]:>8}" for name in class_names)
    print(header)
    for name, row in zip(class_names, cm):
        print(f"{name[:8]:>8} " + " ".join(f"{v:>8}" for v in row))

    print("\nClassification report:")
    print(
        classification_report(
            y_test, preds, labels=range(len(class_names)), target_names=class_names, zero_division=0
        )
    )
    return accuracy, cm


def evaluate(label_map: dict[str, int], best_model_path: Path, test_file: str, title: str) -> None:
    test_data = torch.load(DATA_DIR / test_file)
    evaluate_tensors(best_model_path, test_data["X"], test_data["y"], label_map, title)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-extra", action="store_true", help="baseline : dataset d'origine seul")
    args = parser.parse_args()

    use_extra = not args.no_extra
    best_model_path = MODELS_DIR / ("best_model.pt" if use_extra else "best_model_original.pt")

    label_map = train(use_extra, best_model_path)
    evaluate(label_map, best_model_path, "test_data.pt", "Test d'origine")
    evaluate(label_map, best_model_path, "test_extra_data.pt", "Test sample3 (autre session)")
