from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class GatedAttention(nn.Module):
    def __init__(self, input_dim: int = 384, attention_dim: int = 256, dropout: bool = True):
        super().__init__()
        branch_a = [nn.Linear(input_dim, attention_dim), nn.Tanh()]
        branch_b = [nn.Linear(input_dim, attention_dim), nn.Sigmoid()]
        if dropout:
            branch_a.append(nn.Dropout(0.25))
            branch_b.append(nn.Dropout(0.25))
        self.attention_a = nn.Sequential(*branch_a)
        self.attention_b = nn.Sequential(*branch_b)
        self.attention_c = nn.Linear(attention_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.attention_c(self.attention_a(features) * self.attention_b(features))
        return scores, features


class CLAMSB(nn.Module):
    """Single-branch CLAM classifier matching the released BEPH checkpoints."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 384,
        attention_dim: int = 256,
        dropout: bool = True,
        n_classes: int = 2,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(0.25))
        layers.append(GatedAttention(hidden_dim, attention_dim, dropout))
        self.feature_linear1 = nn.Linear(input_dim, hidden_dim)
        self.attention_net = nn.Sequential(*layers)
        self.classifiers = nn.Linear(hidden_dim, n_classes)
        self.instance_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 2) for _ in range(n_classes)]
        )

    def forward(self, features: torch.Tensor):
        hidden = self.feature_linear1(features.float())
        attention_raw, hidden = self.attention_net(hidden)
        attention_raw = attention_raw.transpose(1, 0)
        attention = F.softmax(attention_raw, dim=1)
        bag_embedding = torch.mm(attention, hidden)
        logits = self.classifiers(bag_embedding)
        probabilities = F.softmax(logits, dim=1)
        prediction = probabilities.argmax(dim=1)
        return logits, probabilities, prediction, attention_raw, {"features": bag_embedding}


def load_checkpoint(model: nn.Module, checkpoint: str | Path, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = OrderedDict()
    for key, value in state.items():
        key = key.replace(".module", "").removeprefix("module.")
        if "instance_loss_fn" not in key:
            cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()

