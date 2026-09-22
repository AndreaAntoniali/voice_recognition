"""Évalue le classifieur et écrit matrices, confidences et prédictions."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import torch

from manual_branch.config import BRANCH_ROOT
from manual_branch.src.checkpoints import load_checkpoint
from manual_branch.src.dataset import SpeakerDataset
from manual_branch.src.decision_classifier import extract_embeddings, load_decision_checkpoint


def write_matrix(path, labels, matrix, digits=6):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_label", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[round(float(value), digits) for value in row]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-checkpoint", type=Path, default=BRANCH_ROOT / "models/best.pt")
    parser.add_argument("--decision-checkpoint", type=Path, default=BRANCH_ROOT / "models/manual_decision.pt")
    parser.add_argument("--test-csv", type=Path, default=BRANCH_ROOT / "data/test.csv")
    parser.add_argument("--root", type=Path, default=BRANCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BRANCH_ROOT / "models/decision_analysis")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    encoder, _, _ = load_checkpoint(args.encoder_checkpoint, args.device)
    classifier, labels, _ = load_decision_checkpoint(args.decision_checkpoint, args.device)
    dataset = SpeakerDataset(args.test_csv, training=False, root=args.root)
    embeddings, true_labels, paths = extract_embeddings(dataset, encoder, args.device)
    mapping = {label: index for index, label in enumerate(labels)}
    targets = torch.tensor([mapping[label] for label in true_labels])
    with torch.inference_mode():
        probabilities = torch.softmax(classifier(embeddings.to(args.device)), dim=1).cpu()
        predictions = probabilities.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    print(f"Accuracy test : {correct}/{len(targets)} = {correct / len(targets):.3f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion = torch.zeros((len(labels), len(labels)), dtype=torch.int64)
    confidence_sum = torch.zeros((len(labels), len(labels)), dtype=torch.float64)
    confidence_count = torch.zeros((len(labels), len(labels)), dtype=torch.int64)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["wav_path", "true_label", "predicted_label", "confidence", "correct"])
        for true, pred, path, probs in zip(true_labels, predictions.tolist(), paths, probabilities):
            true_index = mapping[true]
            confidence = float(probs[pred])
            confusion[true_index, pred] += 1
            confidence_sum[true_index, pred] += confidence
            confidence_count[true_index, pred] += 1
            writer.writerow([path, true, labels[pred], f"{confidence:.8f}", true_index == pred])
    normalized = confusion.float() / confusion.sum(dim=1, keepdim=True).clamp_min(1)
    confidence_mean = confidence_sum / confidence_count.clamp_min(1)
    write_matrix(args.output_dir / "confusion_counts.csv", labels, confusion)
    write_matrix(args.output_dir / "confusion_normalized.csv", labels, normalized)
    write_matrix(args.output_dir / "confidence_mean.csv", labels, confidence_mean)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["label", "support", "correct", "recall", "mean_confidence_correct", "mean_probability_for_true_label"])
        for index, label in enumerate(labels):
            true_mask = targets == index
            correct_mask = true_mask & (predictions == targets)
            row_probabilities = probabilities[true_mask]
            writer.writerow([
                label, int(true_mask.sum()), int(correct_mask.sum()), float(normalized[index, index]),
                float(probabilities[correct_mask, index].mean()) if correct_mask.any() else 0.0,
                float(row_probabilities[:, index].mean()) if len(row_probabilities) else 0.0,
            ])
    print(f"Matrices et prédictions sauvegardées dans : {args.output_dir}")
    counts = Counter((true, labels[pred]) for true, pred in zip(true_labels, predictions.tolist()))
    for true in labels:
        print(true, {pred: counts[(true, pred)] for pred in labels})

    try:
        import matplotlib.pyplot as plt
        plots = [
            ("confusion_matrix.png", "Matrice de confusion", confusion, "d"),
            ("confusion_normalized.png", "Matrice normalisée", normalized, ".2f"),
            ("confidence_mean.png", "Confiance moyenne", confidence_mean, ".2f"),
        ]
        for filename, title, values, fmt in plots:
            figure, axis = plt.subplots(figsize=(7, 6))
            image = axis.imshow(values.numpy(), cmap="Blues", vmin=0, vmax=float(values.max()) or 1)
            figure.colorbar(image, ax=axis)
            axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Prédit", ylabel="Réel", title=title)
            for row in range(len(labels)):
                for column in range(len(labels)):
                    value = float(values[row, column])
                    text = str(int(round(value))) if fmt == "d" else format(value, fmt)
                    axis.text(column, row, text, ha="center", va="center")
            figure.tight_layout()
            figure.savefig(args.output_dir / filename, dpi=160)
            plt.close(figure)
        print("Images PNG générées.")
    except ImportError:
        print("matplotlib absent : les matrices CSV restent disponibles.")


if __name__ == "__main__":
    main()
