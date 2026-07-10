"""
dataset.py
==========
PyTorch Dataset and DataLoader wrappers for the STGNN.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.utils.logger import get_logger

log = get_logger(__name__)


class DroughtDataset(Dataset):
    """
    PyTorch Dataset for the spatio-temporal drought prediction task.

    X: [num_samples, lookback, N, F]
    Y: [num_samples, N, num_targets]
    edge_index: [2, num_edges]
    edge_attr:  [num_edges, num_edge_features]
    """

    def __init__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 edge_index: torch.Tensor,
                 edge_attr:  torch.Tensor,
                 dynamic_weights: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_attr  = edge_attr

        # Optional dynamic edge weights: [num_samples, num_edges]
        if dynamic_weights is not None:
            self.dynamic_weights = torch.tensor(dynamic_weights, dtype=torch.float32)
        else:
            self.dynamic_weights = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        item = {
            "X":          self.X[idx],            # [lookback, N, F]
            "Y":          self.Y[idx],             # [N, num_targets]
            "edge_index": self.edge_index,         # [2, E]
            "edge_attr":  self.edge_attr,          # [E, num_edge_feats]
        }
        if self.dynamic_weights is not None:
            item["edge_weight"] = self.dynamic_weights[idx]  # [E]
        return item


def make_dataloaders(splits: dict,
                     edge_index: torch.Tensor,
                     edge_attr:  torch.Tensor,
                     batch_size: int = 16,
                     num_workers: int = 0) -> dict[str, DataLoader]:
    """Build DataLoaders for all splits."""
    loaders = {}
    for split_name, data in splits.items():
        ds = DroughtDataset(
            X=data["X"], Y=data["Y"],
            edge_index=edge_index, edge_attr=edge_attr
        )
        loaders[split_name] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        log.info(f"DataLoader [{split_name}]: {len(ds)} samples, "
                 f"batch_size={batch_size}")
    return loaders
