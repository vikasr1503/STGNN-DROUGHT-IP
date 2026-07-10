"""
callbacks.py
============
Training callbacks: checkpoint saving, LR scheduling, metric tracking.
"""

from pathlib import Path
import torch
import torch.nn as nn
from src.utils.logger import get_logger

log = get_logger(__name__)


class ModelCheckpoint:
    """Save best model weights based on monitored metric."""

    def __init__(self, save_dir: str, run_name: str,
                 monitor: str = "val_loss", mode: str = "min"):
        self.save_path = Path(save_dir) / f"{run_name}_best.pth"
        self.monitor   = monitor
        self.best      = float("inf") if mode == "min" else -float("inf")
        self.mode      = mode

    def __call__(self, model: nn.Module, metric_val: float) -> bool:
        """Returns True if new best was saved."""
        improved = (self.mode == "min" and metric_val < self.best) or \
                   (self.mode == "max" and metric_val > self.best)
        if improved:
            self.best = metric_val
            torch.save(model.state_dict(), self.save_path)
            log.info(f"Checkpoint saved ({self.monitor}={metric_val:.4f}) → {self.save_path}")
            return True
        return False

    def load_best(self, model: nn.Module, device: torch.device) -> nn.Module:
        model.load_state_dict(torch.load(self.save_path, map_location=device))
        log.info(f"Loaded best weights from {self.save_path}")
        return model


class MetricsTracker:
    """Track and log train/val metrics per epoch."""

    def __init__(self):
        self.history = {}

    def update(self, epoch: int, **kwargs) -> None:
        for k, v in kwargs.items():
            self.history.setdefault(k, []).append(v)
        msg = "  ".join(f"{k}={v:.4f}" for k, v in kwargs.items())
        log.info(f"Epoch {epoch:4d} | {msg}")

    def get(self, key: str) -> list:
        return self.history.get(key, [])

    def best(self, key: str, mode: str = "min") -> float:
        vals = self.get(key)
        return min(vals) if mode == "min" else max(vals)
