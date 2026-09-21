"""Diagnostic de mémorisation sur 8 à 16 fichiers, sans mesure de généralisation."""
import argparse
from pathlib import Path
import tempfile

import torch

from manual_branch.config import AudioConfig, BRANCH_ROOT
from manual_branch.src.checkpoints import load_checkpoint, save_checkpoint
from manual_branch.src.dataset import SpeakerDataset
from manual_branch.src.manual_encoder import ManualEncoder
from manual_branch.src.synthetic import create_synthetic_dataset
from manual_branch.src.training import make_loader, run_epoch, seed_everything


def run_overfit(
    csv_path, *, root=BRANCH_ROOT, output_dir=BRANCH_ROOT / "models/overfit",
    count=8, epochs=300, learning_rate=0.003, device="cpu", seed=42,
    max_loss=0.05, min_cosine=0.94, max_loss_ratio=0.2,
):
    if not 8 <= count <= 16 or epochs < 1:
        raise ValueError("Le diagnostic exige 8 à 16 fichiers et au moins une époque.")
    seed_everything(seed)
    dataset = SpeakerDataset(csv_path, training=True, root=root)
    if len(dataset) < count:
        raise ValueError(f"Il faut {count} fichiers ; le CSV n'en contient que {len(dataset)}.")
    selected = torch.randperm(len(dataset))[:count].tolist()
    if len({dataset.records[index]["wav_path"] for index in selected}) != count:
        raise ValueError("Le sous-ensemble de surapprentissage doit contenir des WAV distincts.")
    # Un crop aléatoire par fichier, puis figé pour isoler la capacité à mémoriser.
    cached = [dataset[index] for index in selected]
    loader = make_loader(cached, count, shuffle=False, seed=seed)
    model = ManualEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
    initial = run_epoch(model, loader, device)
    best = float("inf")
    checkpoint_path = Path(output_dir) / "best.pt"
    for epoch in range(1, epochs + 1):
        run_epoch(model, loader, device, optimizer)
        metrics = run_epoch(model, loader, device)
        if metrics["loss_total"] < best:
            best = metrics["loss_total"]
            save_checkpoint(checkpoint_path, model, AudioConfig(), epoch, best, optimizer)
        success = (
            metrics["loss_total"] <= max_loss
            and 1 - metrics["loss_cosine"] >= min_cosine
            and metrics["loss_total"] <= initial["loss_total"] * max_loss_ratio
        )
        if epoch == 1 or epoch % 10 == 0 or success or epoch == epochs:
            print(f"Overfit {epoch}/{epochs} | perte={metrics['loss_total']:.6f} | cosinus={1 - metrics['loss_cosine']:.6f}", flush=True)
        if success:
            break
    # Vérifier le checkpoint réellement relu, pas seulement le modèle en mémoire.
    restored, _, _ = load_checkpoint(checkpoint_path, device)
    final = run_epoch(restored, loader, device)
    passed = (
        final["loss_total"] <= max_loss
        and 1 - final["loss_cosine"] >= min_cosine
        and final["loss_total"] <= initial["loss_total"] * max_loss_ratio
    )
    print(f"Perte initiale={initial['loss_total']:.6f} ; meilleure={final['loss_total']:.6f} ; réussite={passed}")
    if not passed:
        raise RuntimeError("Échec du test de surapprentissage : seuils non atteints. Vérifiez les couples WAV/ECAPA ou augmentez les époques.")
    return {"initial": initial, "final": final, "epoch": epoch, "checkpoint": str(checkpoint_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, default=BRANCH_ROOT / "data/train.csv")
    source.add_argument("--synthetic", action="store_true", help="Signaux et cibles artificiels, sans ECAPA réel")
    parser.add_argument("--root", type=Path, default=BRANCH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BRANCH_ROOT / "models/overfit")
    parser.add_argument("--count", type=int, choices=range(8, 17), default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads doit être >= 1")
    torch.set_num_threads(args.threads)
    kwargs = dict(root=args.root, output_dir=args.output_dir, count=args.count, epochs=args.epochs,
                  learning_rate=args.learning_rate, device=args.device, seed=args.seed)
    try:
        if args.synthetic:
            print("DIAGNOSTIC SYNTHÉTIQUE : les cibles ne sont pas des embeddings ECAPA réels.")
            with tempfile.TemporaryDirectory(prefix="manual-overfit-") as temporary:
                run_overfit(create_synthetic_dataset(temporary, args.count), **kwargs)
        else:
            run_overfit(args.csv, **kwargs)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
