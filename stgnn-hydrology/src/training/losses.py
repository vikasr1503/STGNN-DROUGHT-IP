"""
losses.py
=========
Loss functions for drought prediction.
"""
import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """MSE that ignores NaN target values (missing observations)."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        return ((pred[mask] - target[mask]) ** 2).mean()


class MaskedMAELoss(nn.Module):
    """MAE that ignores NaN target values."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)
        return (pred[mask] - target[mask]).abs().mean()


class CombinedLoss(nn.Module):
    """alpha * MSE + (1-alpha) * MAE"""

    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = MaskedMSELoss()
        self.mae = MaskedMAELoss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + \
               (1 - self.alpha) * self.mae(pred, target)
