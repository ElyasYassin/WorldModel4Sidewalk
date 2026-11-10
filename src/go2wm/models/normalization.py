from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Normalizer(nn.Module):
    """Per-dimension standardization; statistics fit on the training split only and saved as buffers."""

    def __init__(self, dim: int, eps: float = 1e-6, min_std: float = 1e-3):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = eps
        self.min_std = min_std

    def fit(self, x: np.ndarray | torch.Tensor) -> "Normalizer":
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(-1, self.mean.numel())
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp_min(self.min_std))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean
