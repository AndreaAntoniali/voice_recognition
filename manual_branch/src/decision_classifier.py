from pathlib import Path
import json

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from manual_branch.src.checkpoints import load_checkpoint
from manual_branch.src.dataset import SpeakerDataset


class DecisionClassifier(nn.Module):
    """Classifieur séparé : embedding 192D -> identité du locuteur."""

    def __init__(self, num_classes: int):
        if num_classes < 2:
            raise ValueError("Il faut au moins deux locuteurs.")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(192, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != 192:
            raise ValueError("Le classifieur attend [batch, 192].")
        return self.network(embedding.float())


@torch.inference_mode()
def extract_embeddings(dataset: SpeakerDataset, encoder, device="cpu"):
    encoder.eval().to(device)
    embeddings, labels, paths = [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        embedding = encoder(item["log_mel"].unsqueeze(0).to(device))[0].cpu()
        embeddings.append(embedding)
        labels.append(item["label"])
        paths.append(item["wav_path"])
    if not embeddings:
        raise ValueError("Aucun fichier disponible pour extraire les embeddings.")
    return torch.stack(embeddings), labels, paths


def make_embedding_dataset(embeddings, labels, label_to_index):
    targets = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)
    return TensorDataset(embeddings.float(), targets)


def save_decision_checkpoint(path, classifier, labels, encoder_checkpoint, epoch, validation_accuracy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": 1,
        "architecture": "decision_mlp_192_128_64",
        "model_state_dict": classifier.state_dict(),
        "labels": list(labels),
        "encoder_checkpoint": str(encoder_checkpoint),
        "epoch": int(epoch),
        "validation_accuracy": float(validation_accuracy),
    }, path)


def load_decision_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    labels = checkpoint.get("labels")
    if checkpoint.get("version") != 1 or checkpoint.get("architecture") != "decision_mlp_192_128_64" or not labels:
        raise ValueError(f"Checkpoint classifieur invalide : {path}")
    classifier = DecisionClassifier(len(labels))
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return classifier.to(device).eval(), labels, checkpoint


def run_classifier_epoch(classifier, loader, device="cpu", optimizer=None):
    training = optimizer is not None
    classifier.train(training)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(training):
        for embeddings, targets in loader:
            embeddings, targets = embeddings.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = classifier(embeddings)
            loss = nn.functional.cross_entropy(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += len(targets)
    if not count:
        raise ValueError("Dataset de classification vide.")
    return {"loss": total_loss / count, "accuracy": correct / count}
