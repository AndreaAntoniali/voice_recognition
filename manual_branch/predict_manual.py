"""Prédit le locuteur avec la branche manuelle uniquement."""
import argparse
from pathlib import Path
import torch

from manual_branch.config import BRANCH_ROOT
from manual_branch.src.decision_classifier import load_decision_checkpoint
from manual_branch.src.inference import WavEncoder


def predict(wav_path, encoder_checkpoint=BRANCH_ROOT / "models/best.pt", decision_checkpoint=BRANCH_ROOT / "models/manual_decision.pt", device="cpu"):
    encoder = WavEncoder(encoder_checkpoint, device=device)
    classifier, labels, _ = load_decision_checkpoint(decision_checkpoint, device)
    with torch.inference_mode():
        embedding = encoder.encode_wav(wav_path).to(device)
        probabilities = torch.softmax(classifier(embedding.unsqueeze(0))[0], dim=0)
        index = int(probabilities.argmax())
    return labels[index], float(probabilities[index]), {label: float(probabilities[i]) for i, label in enumerate(labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path, default=BRANCH_ROOT / "models/best.pt")
    parser.add_argument("--decision-checkpoint", type=Path, default=BRANCH_ROOT / "models/manual_decision.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    label, confidence, probabilities = predict(args.wav_path, args.encoder_checkpoint, args.decision_checkpoint, args.device)
    print(f"Locuteur prédit par la branche manuelle : {label}")
    print(f"Confiance : {confidence:.4f}")
    for name, value in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
