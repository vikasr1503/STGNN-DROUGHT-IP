"""
adjacency.py
============
Builds the adjacency matrix and edge_index tensor for PyTorch Geometric.

Supports:
  - Static adjacency (binary river connectivity)
  - Dynamic adjacency (time-varying edge weights)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.utils.logger import get_logger

log = get_logger(__name__)

GRAPH_DIR = Path("data/graph")


def build_static_adjacency(edges: pd.DataFrame,
                            n_nodes: int = 130,
                            self_loops: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build PyG-compatible edge_index and edge_attr tensors.

    Returns:
      edge_index: LongTensor [2, num_edges]  (src, dst pairs)
      edge_attr:  FloatTensor [num_edges, num_edge_features]
    """
    src = torch.tensor(edges["src_idx"].values, dtype=torch.long)
    dst = torch.tensor(edges["dst_idx"].values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)

    # Static edge features
    feat_cols = [c for c in ["elevation_difference_m", "river_order",
                              "flow_accumulation", "river_length_km",
                              "travel_distance_km"]
                 if c in edges.columns]
    if feat_cols:
        edge_attr = torch.tensor(
            edges[feat_cols].fillna(0).values, dtype=torch.float32
        )
    else:
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float32)

    if self_loops:
        loop_idx   = torch.arange(n_nodes, dtype=torch.long)
        loop_edge  = torch.stack([loop_idx, loop_idx], dim=0)
        loop_attr  = torch.zeros(n_nodes, edge_attr.shape[1], dtype=torch.float32)
        edge_index = torch.cat([edge_index, loop_edge], dim=1)
        edge_attr  = torch.cat([edge_attr,  loop_attr],  dim=0)

    log.info(f"edge_index shape: {edge_index.shape}  "
             f"edge_attr shape: {edge_attr.shape}")

    # Save for inspection
    np.save(GRAPH_DIR / "edge_index.npy", edge_index.numpy())
    np.save(GRAPH_DIR / "edge_attr.npy",  edge_attr.numpy())

    return edge_index, edge_attr


def build_adjacency_matrix(edges: pd.DataFrame,
                            n_nodes: int = 130,
                            weighted: bool = False,
                            weight_col: str = "weight") -> np.ndarray:
    """
    Build dense N×N adjacency matrix.
    weighted=True uses the edge weight column.
    """
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in edges.iterrows():
        i, j = int(row["src_idx"]), int(row["dst_idx"])
        if 0 <= i < n_nodes and 0 <= j < n_nodes:
            val = float(row[weight_col]) if weighted and weight_col in row else 1.0
            A[i, j] = val
    return A


def build_dynamic_edge_weights_sequence(edges: pd.DataFrame,
                                        features: pd.DataFrame,
                                        dates: pd.DatetimeIndex,
                                        compute_fn) -> np.ndarray:
    """
    Build dynamic edge weight matrix for a full time sequence.

    Returns: array of shape [T, num_edges]
    """
    from src.graph.build_edges import compute_dynamic_edge_weights
    T = len(dates)
    E = len(edges)
    weights = np.zeros((T, E), dtype=np.float32)
    for t, date in enumerate(dates):
        weights[t] = compute_dynamic_edge_weights(edges, features, date)
    log.info(f"Dynamic edge weights: shape {weights.shape}")
    np.save(GRAPH_DIR / "dynamic_edge_weights.npy", weights)
    return weights
