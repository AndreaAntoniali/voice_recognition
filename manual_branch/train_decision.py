"""Entraîne le classifieur sur les embeddings du CNN manuel."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from manual_branch.config import BRANCH_ROOT
from manual_branch.src.checkpoints import load_checkpoint
from manual_branch.src.dataset import SpeakerDataset
from manual_branch.src.decision_classifier import (
    DecisionClassifier, extract_embeddings, make_embedding_dataset,
    run_classifier_epoch, save_decision_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", type=Path, default=BRANCH_ROOT / "models/best.pt")
    parser.add_argument("--train-csv", type=Path, default=BRANCH_ROOT / "data/train.csv")
    parser.add_argument("--validation-csv", type=Path, default=BRANCH_ROOT / "data/validation.csv")
    parser.add_argument("--root", type=Path, default=BRANCH_ROOT)
    parser.add_argument("--output", type=Path, default=BRANCH_ROOT / "models/manual_decision.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    encoder, _, _ = load_checkpoint(args.encoder_checkpoint, args.device)
    train_set = SpeakerDataset(args.train_csv, training=False, root=args.root)
    validation_set = SpeakerDataset(args.validation_csv, training=False, root=args.root)
    train_embeddings, train_labels, _ = extract_embeddings(train_set, encoder, args.device)
    val_embeddings, val_labels, _ = extract_embeddings(validation_set, encoder, args.device)
    labels = sorted(set(train_labels) | set(val_labels))
    label_to_index = {label: index for index, label in enumerate(labels)}
    train_loader = DataLoader(make_embedding_dataset(train_embeddings, train_labels, label_to_index), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(make_embedding_dataset(val_embeddings, val_labels, label_to_index), batch_size=args.batch_size, shuffle=False)
    classifier = DecisionClassifier(len(labels)).to(args.device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train = run_classifier_epoch(classifier, train_loader, args.device, optimizer)
        validation = run_classifier_epoch(classifier, val_loader, args.device)
        if validation["accuracy"] >= best_accuracy:
            best_accuracy = validation["accuracy"]
            save_decision_checkpoint(args.output, classifier, labels, args.encoder_checkpoint, epoch, best_accuracy)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"Époque {epoch}/{args.epochs} | train accuracy={train['accuracy']:.3f} | validation accuracy={validation['accuracy']:.3f}", flush=True)
    print(f"Classifieur sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
