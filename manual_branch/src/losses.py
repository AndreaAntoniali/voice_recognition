import torch
import torch.nn.functional as F


def supervised_contrastive_loss(embedding: torch.Tensor, labels: list[str] | tuple[str, ...], temperature: float = 0.07) -> dict:
    """Perte contrastive supervisée : positifs = même locuteur, négatifs = autres."""
    if embedding.ndim != 2 or embedding.shape[1] != 192 or len(labels) != embedding.shape[0]:
        raise ValueError("Embeddings [batch, 192] et labels de même longueur requis.")
    if embedding.shape[0] < 2 or temperature <= 0:
        raise ValueError("Il faut au moins deux exemples et une température positive.")
    z = F.normalize(embedding, dim=1)
    logits = z @ z.T / temperature
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, -torch.inf)
    label_tensor = torch.tensor([hash(str(label)) for label in labels], device=z.device)
    positives = (label_tensor[:, None] == label_tensor[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        raise ValueError("Chaque batch doit contenir au moins une paire du même locuteur.")
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    loss = -(log_prob.masked_fill(~positives, 0).sum(dim=1) / positive_count)[valid].mean()
    cosine = ((z[:, None] * z[None, :]).sum(-1)[positives]).mean()
    return {"loss_total": loss, "loss_cosine": 1 - cosine, "loss_mse": torch.zeros_like(loss)}
