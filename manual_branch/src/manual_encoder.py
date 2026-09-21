import torch
from torch import nn
import torch.nn.functional as F

from manual_branch.config import EMBEDDING_DIM


class ManualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        for incoming, outgoing in ((1, 16), (16, 32), (32, 128)):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, outgoing), nn.ReLU(), nn.MaxPool2d(2),
            ))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, EMBEDDING_DIM))

    def forward(self, log_mel: torch.Tensor, trace: bool = False) -> torch.Tensor:
        if log_mel.ndim != 3 or log_mel.shape[1] != 80 or log_mel.shape[2] < 8:
            raise ValueError("Entrée CNN attendue : [batch, 80, temps], avec temps >= 8.")

        def show(name, value):
            if trace:
                print(f"{name}: {list(value.shape)} ({value.dtype})")

        show("Log-Mel normalisé", log_mel)
        x = log_mel.unsqueeze(1)
        show("Ajout du canal", x)
        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            show(f"Bloc CNN {index}", x)
        x = self.pool(x)
        show("AdaptiveAvgPool2d", x)
        x = x.flatten(1)
        show("Vecteur CNN", x)
        for index, layer in enumerate(self.projection):
            x = layer(x)
            show(f"Projection {index + 1}", x)
        # Un vecteur nul ne peut pas être normalisé : signaler un modèle dégénéré.
        if not torch.isfinite(x).all() or torch.any(x.norm(dim=-1) <= 1e-12):
            raise ValueError("Projection non finie ou de norme nulle ; vérifiez le modèle et les entrées.")
        x = F.normalize(x, p=2, dim=-1)
        show("Embedding L2", x)
        return x
