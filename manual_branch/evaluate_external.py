"""Évalue la branche manuelle sur un dossier externe, sans modifier les CSV."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from manual_branch.config import BRANCH_ROOT
from manual_branch.src.checkpoints import load_checkpoint
from manual_branch.src.decision_classifier import load_decision_checkpoint
from manual_branch.src.log_mel import LogMelExtractor
from manual_branch.src.audio import adjust_duration, load_wav


@torch.inference_mode()
def evaluate(audio_dir, encoder_checkpoint, decision_checkpoint, device="cpu", expected_label=None):
    audio_dir = Path(audio_dir)
    encoder, config, _ = load_checkpoint(encoder_checkpoint, device)
    classifier, labels, _ = load_decision_checkpoint(decision_checkpoint, device)
    extractor = LogMelExtractor(config).to(device).eval()
    # Accepte aussi une organisation supplémentaire, par exemple
    # external_audio/nabil-2/nabil2/*.wav.
    files = sorted(path for path in audio_dir.rglob("*.wav"))
    if not files:
        raise ValueError(f"Aucun fichier WAV trouvé dans {audio_dir}/<locuteur>/.")
    results = []
    for path in files:
        relative_parts = path.relative_to(audio_dir).parts
        expected = expected_label or (relative_parts[0] if len(relative_parts) > 2 else path.parent.name)
        waveform = adjust_duration(load_wav(path, config), training=False, config=config)
        features = extractor(waveform.to(device))
        embedding = encoder(features.unsqueeze(0))[0]
        probabilities = torch.softmax(classifier(embedding.unsqueeze(0))[0], dim=0)
        index = int(probabilities.argmax())
        predicted = labels[index]
        confidence = float(probabilities[index])
        results.append((expected, predicted, confidence, path))
        print(f"{path}: attendu={expected}, prédit={predicted}, confiance={confidence:.3f}")
    correct = sum(expected == predicted for expected, predicted, _, _ in results)
    print(f"Accuracy externe : {correct}/{len(results)} = {correct / len(results):.3f}")
    confusion = Counter((expected, predicted) for expected, predicted, _, _ in results)
    for expected in sorted({item[0] for item in results}):
        print(expected, {predicted: confusion[(expected, predicted)] for predicted in labels})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_dir", type=Path, help="Dossier contenant un sous-dossier par locuteur")
    parser.add_argument("--encoder-checkpoint", type=Path, default=BRANCH_ROOT / "models/best.pt")
    parser.add_argument("--decision-checkpoint", type=Path, default=BRANCH_ROOT / "models/manual_decision.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-label", default=None, help="Label réel, par exemple nabil si le dossier s'appelle nabil-2")
    args = parser.parse_args()
    evaluate(args.audio_dir, args.encoder_checkpoint, args.decision_checkpoint, args.device, args.expected_label)


if __name__ == "__main__":
    main()
