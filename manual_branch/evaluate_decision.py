"""Évalue le classifieur manuel sur test.csv."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from manual_branch.config import BRANCH_ROOT
from manual_branch.src.checkpoints import load_checkpoint
from manual_branch.src.dataset import SpeakerDataset
from manual_branch.src.decision_classifier import extract_embeddings, load_decision_checkpoint, make_embedding_dataset, run_classifier_epoch
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", type=Path, default=BRANCH_ROOT / "models/best.pt")
    parser.add_argument("--decision-checkpoint", type=Path, default=BRANCH_ROOT / "models/manual_decision.pt")
    parser.add_argument("--test-csv", type=Path, default=BRANCH_ROOT / "data/test.csv")
    parser.add_argument("--root", type=Path, default=BRANCH_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    encoder, _, _ = load_checkpoint(args.encoder_checkpoint, args.device)
    classifier, labels, _ = load_decision_checkpoint(args.decision_checkpoint, args.device)
    dataset = SpeakerDataset(args.test_csv, training=False, root=args.root)
    embeddings, true_labels, paths = extract_embeddings(dataset, encoder, args.device)
    mapping = {label: index for index, label in enumerate(labels)}
    targets = torch.tensor([mapping[label] for label in true_labels])
    with torch.inference_mode():
        predictions = classifier(embeddings.to(args.device)).argmax(dim=1).cpu()
    correct = (predictions == targets).sum().item()
    print(f"Accuracy test : {correct}/{len(targets)} = {correct / len(targets):.3f}")
    confusion = Counter((true, labels[pred]) for true, pred in zip(true_labels, predictions.tolist()))
    for true in labels:
        print(true, {pred: confusion[(true, pred)] for pred in labels})


if __name__ == "__main__":
    main()
