from __future__ import annotations

import torch
import torch.nn.functional as F


def rollout_loss(pred_norm: torch.Tensor, target_norm: torch.Tensor, rollout_weight: float, huber_delta: float = 1.0) -> torch.Tensor:
    """Huber on normalized deltas. Step 0 is the teacher-forced one-step term (weight 1);
    later steps are open-loop autoregressive terms weighted by rollout_weight."""
    per_step = F.huber_loss(pred_norm, target_norm, delta=huber_delta, reduction="none").mean(dim=(0, 2))  # [H]
    w = torch.full_like(per_step, rollout_weight)
    w[0] = 1.0
    return (w * per_step).sum() / w.sum()
