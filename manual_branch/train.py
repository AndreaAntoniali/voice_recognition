"""Depuis la racine : python -m manual_branch.train --help."""
import argparse
from pathlib import Path

import torch

from manual_branch.config import BRANCH_ROOT
from manual_branch.src.dataset import SpeakerDataset
from manual_branch.src.training import fit


def main():
    parser = argparse.ArgumentParser(description="Distillation ECAPA -> encodeur manuel 192D")
    parser.add_argument("--train-csv", type=Path, default=BRANCH_ROOT / "data/train.csv")
    parser.add_argument("--validation-csv", type=Path, default=BRANCH_ROOT / "data/validation.csv")
    parser.add_argument("--root", type=Path, default=BRANCH_ROOT, help="Base des chemins contenus dans les CSV")
    parser.add_argument("--output-dir", type=Path, default=BRANCH_ROOT / "models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads doit être >= 1")
    torch.set_num_threads(args.threads)
    try:
        train = SpeakerDataset(args.train_csv, training=True, root=args.root)
        validation = SpeakerDataset(args.validation_csv, training=False, root=args.root)
        overlap = {record["wav_path"] for record in train.records} & {record["wav_path"] for record in validation.records}
        if overlap:
            raise ValueError(f"WAV présents dans train et validation : {sorted(overlap)[:3]}")
        fit(
            train, validation, args.output_dir, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, device=args.device,
            workers=args.workers, seed=args.seed,
        )
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Meilleur checkpoint : {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
