"""
trainer.py
==========
Training loop for the STGNN model.
Handles:
  - Forward pass + loss
  - Backpropagation
  - Validation
  - Early stopping
  - Checkpoint saving
  - TensorBoard logging
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.training.losses import CombinedLoss
from src.utils.logger import get_logger

log = get_logger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = float("inf")
        self.stop      = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train(model:       nn.Module,
          loaders:     dict,
          cfg:         dict,
          edge_index:  torch.Tensor,
          device:      torch.device,
          run_name:    str = "gcn_gru") -> dict:
    """
    Full training loop.

    model:      DynamicDirectedSTGNN or GATGRUModel
    loaders:    {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    cfg:        training config dict (from configs/train.yaml)
    edge_index: [2, E] edge connectivity
    device:     torch.device
    run_name:   used for checkpoint and log naming

    Returns dict with training history.
    """
    train_cfg = cfg["training"]
    save_dir  = Path(train_cfg["save_dir"])
    log_dir   = Path(train_cfg["log_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    edge_index = edge_index.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    criterion = CombinedLoss(alpha=0.7)
    stopper   = EarlyStopping(patience=train_cfg["patience"])
    writer    = SummaryWriter(log_dir=str(log_dir))

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in loaders["train"]:
            X          = batch["X"].to(device)         # [B, lookback, N, F]
            Y          = batch["Y"].to(device)         # [B, N, num_targets]
            ew         = batch.get("edge_weight")
            if ew is not None:
                ew = ew.to(device)

            optimizer.zero_grad()
            pred = model(X, edge_index, edge_weight=ew)
            loss = criterion(pred, Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loaders["train"])
        scheduler.step()

        # ── Validate ───────────────────────────────────────────────────────
        val_loss = _evaluate(model, loaders["val"], criterion, edge_index, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        if epoch % 10 == 0 or epoch == 1:
            log.info(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f}")

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_dir / f"{run_name}_best.pth")

        if stopper(val_loss):
            log.info(f"Early stopping at epoch {epoch}.")
            break

    writer.close()
    log.info(f"Training complete. Best val loss: {best_val:.4f}")
    return history


def _evaluate(model, loader, criterion, edge_index, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            X  = batch["X"].to(device)
            Y  = batch["Y"].to(device)
            ew = batch.get("edge_weight")
            if ew is not None:
                ew = ew.to(device)
            pred = model(X, edge_index, edge_weight=ew)
            total += criterion(pred, Y).item()
    return total / max(len(loader), 1)
