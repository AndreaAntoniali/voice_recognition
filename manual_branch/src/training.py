import csv
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from manual_branch.config import AudioConfig
from manual_branch.src.checkpoints import save_checkpoint
from manual_branch.src.losses import supervised_contrastive_loss
from manual_branch.src.manual_encoder import ManualEncoder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, batch_size, shuffle, workers=0, seed=42):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
    )


def run_epoch(model, loader, device="cpu", optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"loss_total": 0.0, "loss_cosine": 0.0, "loss_mse": 0.0}
    count = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            features = batch["log_mel"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            losses = supervised_contrastive_loss(model(features), batch["label"])
            if not all(torch.isfinite(value) for value in losses.values()):
                raise ValueError(f"Perte non finie pour {batch['wav_path']}")
            if training:
                losses["loss_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
            size = features.shape[0]
            for key in sums:
                sums[key] += losses[key].detach().item() * size
            count += size
    if count == 0:
        raise ValueError("Impossible d'évaluer ou d'entraîner sur un dataset vide.")
    return {key: value / count for key, value in sums.items()}


def fit(
    train_dataset, validation_dataset, output_dir, *, epochs=50, batch_size=16,
    learning_rate=1e-3, weight_decay=1e-4, device="cpu", workers=0, seed=42,
    config=AudioConfig(),
):
    if epochs < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs et batch_size doivent être positifs ; workers doit être >= 0.")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("learning_rate doit être fini et positif ; weight_decay fini et >= 0.")
    seed_everything(seed)
    model = ManualEncoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = make_loader(train_dataset, batch_size, True, workers, seed)
    validation_loader = make_loader(validation_dataset, batch_size, False, workers, seed)
    model.eval()
    with torch.no_grad():
        model(validation_dataset[0]["log_mel"].unsqueeze(0).to(device), trace=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        keys = ["epoch"] + [f"{split}_{key}" for split in ("train", "validation") for key in ("loss_total", "loss_cosine", "loss_mse")]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train = run_epoch(model, train_loader, device, optimizer)
            validation = run_epoch(model, validation_loader, device)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()}, **{f"validation_{k}": v for k, v in validation.items()}}
            writer.writerow(row)
            stream.flush()
            history.append(row)
            improved = validation["loss_total"] < best_loss
            if improved:
                best_loss = validation["loss_total"]
                save_checkpoint(output_dir / "best.pt", model, config, epoch, best_loss, optimizer)
            print(
                f"Époque {epoch}/{epochs} | train={train['loss_total']:.6f} | "
                f"validation={validation['loss_total']:.6f} | "
                f"cosinus validation={1 - validation['loss_cosine']:.6f}"
                + (" | meilleur checkpoint sauvegardé" if improved else ""), flush=True,
            )
    return history
