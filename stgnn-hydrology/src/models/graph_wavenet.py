"""
graph_wavenet.py
================
Graph WaveNet — Phase 3 Architecture
Adaptive Graph Learning + Dilated Causal Convolutions

Reference: Wu et al., "Graph WaveNet for Deep Spatial-Temporal
Graph Modeling", IJCAI 2019.

Key innovations over GCN+GRU:
  1. Learnable adaptive adjacency matrix (no fixed graph needed)
  2. Dilated causal convolutions replace GRU (faster, parallelisable)
  3. Stacked WaveNet residual blocks capture multi-scale temporal patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class AdaptiveAdjacency(nn.Module):
    """
    Learns a node embedding E ∈ R^{N×d} and computes
    A_adaptive = SoftMax(ReLU(E @ E^T))
    allowing the model to discover hidden spatial dependencies.
    """
    def __init__(self, n_nodes: int, embed_dim: int = 10):
        super().__init__()
        self.E1 = nn.Embedding(n_nodes, embed_dim)
        self.E2 = nn.Embedding(n_nodes, embed_dim)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """Returns [N, N] adaptive adjacency matrix."""
        e1 = self.E1(node_ids)    # [N, d]
        e2 = self.E2(node_ids)    # [N, d]
        A  = F.relu(torch.mm(e1, e2.T))
        A  = F.softmax(A, dim=1)
        return A


class WaveNetBlock(nn.Module):
    """
    Single dilated causal convolution + GCN residual block.
    """
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 kernel_size:  int = 2,
                 dilation:     int = 1,
                 dropout:      float = 0.3):
        super().__init__()

        self.filter_conv = nn.Conv1d(in_channels, out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=0)
        self.gate_conv   = nn.Conv1d(in_channels, out_channels,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=0)

        self.gcn    = GCNConv(out_channels, out_channels)
        self.bn     = nn.BatchNorm1d(out_channels)
        self.skip   = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B*N, C, T]
        Returns: residual [B*N, C, T'], skip [B*N, C, T']
        """
        residual = x

        # Gated dilated convolution
        h_filter = torch.tanh(self.filter_conv(x))
        h_gate   = torch.sigmoid(self.gate_conv(x))
        h        = h_filter * h_gate          # [B*N, out_channels, T']
        h        = F.dropout(h, p=self.dropout, training=self.training)

        # GCN at final timestep of each block
        h_last = h[:, :, -1]                  # [B*N, out_channels]
        h_last = self.gcn(h_last, edge_index, edge_weight)
        h_last = F.relu(h_last)
        h      = h + h_last.unsqueeze(-1)     # broadcast back to time

        # Skip connection
        skip = h

        # Residual connection (align channels)
        T_new = h.shape[-1]
        T_old = residual.shape[-1]
        if T_old > T_new:
            residual = residual[:, :, T_old - T_new:]
        residual = self.skip(residual) + h

        return self.bn(residual), skip


class GraphWaveNet(nn.Module):
    """
    Full Graph WaveNet model for drought prediction.

    Architecture:
      Input projection
      → N × WaveNetBlock (dilated convolutions + GCN)
      → Skip connection sum
      → Dense output head
    """

    def __init__(self,
                 in_channels:   int   = 11,
                 hidden:        int   = 32,
                 n_nodes:       int   = 130,
                 num_blocks:    int   = 4,
                 kernel_size:   int   = 2,
                 dropout:       float = 0.3,
                 num_targets:   int   = 3,
                 adaptive_dim:  int   = 10):
        super().__init__()

        self.adaptive_adj = AdaptiveAdjacency(n_nodes, adaptive_dim)
        self.node_ids     = nn.Parameter(
            torch.arange(n_nodes), requires_grad=False
        )

        self.input_proj = nn.Conv1d(in_channels, hidden, kernel_size=1)

        # Stacked WaveNet blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(
                WaveNetBlock(hidden, hidden, kernel_size, dilation, dropout)
            )

        self.skip_proj = nn.Conv1d(hidden * num_blocks, hidden * 2, kernel_size=1)

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(hidden * 2, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, num_targets, kernel_size=1),
        )

    def forward(self,
                X:           torch.Tensor,
                edge_index:  torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        X:          [B, lookback, N, F]
        edge_index: [2, E]
        Returns:    [B, N, num_targets]
        """
        B, T, N, F = X.shape

        # Reshape to [B*N, F, T] for Conv1d
        x = X.permute(0, 2, 3, 1)        # [B, N, F, T]
        x = x.reshape(B * N, F, T)

        x = self.input_proj(x)            # [B*N, hidden, T]

        # Adaptive adjacency
        A_adapt  = self.adaptive_adj(self.node_ids)  # [N, N]
        # Convert dense adaptive adjacency to sparse edge_index for GCN
        adapt_ei, adapt_ew = _dense_to_sparse(A_adapt, B)

        skips = []
        for block in self.blocks:
            x, skip = block(x, adapt_ei, adapt_ew)
            skips.append(skip[:, :, -1:])   # keep last timestep

        # Aggregate skips
        skip_cat = torch.cat(skips, dim=1)                  # [B*N, H*blocks, 1]
        out = self.skip_proj(skip_cat)                       # [B*N, H*2, 1]
        out = self.head(out).squeeze(-1)                     # [B*N, num_targets]
        out = out.reshape(B, N, -1)                          # [B, N, num_targets]
        return out


def _dense_to_sparse(A: torch.Tensor, batch_size: int):
    """Convert dense [N, N] matrix to batched sparse edge_index."""
    idx = (A > 0).nonzero(as_tuple=False).T   # [2, E]
    ew  = A[idx[0], idx[1]]
    return idx, ew
