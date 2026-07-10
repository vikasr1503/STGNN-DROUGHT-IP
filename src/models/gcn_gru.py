"""
gcn_gru.py
==========
Dynamic Directed STGNN — Phase 1 Architecture
  GCN  (spatial)  → GRU (temporal) → Dense (output)

Input per timestep: [N, F]  (N sub-basins, F features)
Full sequence:      [batch, lookback, N, F]
Output:             [batch, N, num_targets]  (SRI-3, SRI-6, SRI-12)

Reference architecture follows the methodology from:
"Physics-Informed Dynamic Directed Graph Neural Network
 for Hydrological Drought Prediction in Krishna River Basin"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class TemporalGCN(nn.Module):
    """
    Apply GCN at each timestep, producing a latent spatial embedding.
    """

    def __init__(self,
                 in_channels:     int,
                 hidden_channels: int,
                 num_layers:      int = 2,
                 dropout:         float = 0.3,
                 normalize:       bool = True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels,
                                  normalize=normalize))
        self.bns.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels,
                                      normalize=normalize))
            self.bns.append(BatchNorm(hidden_channels))

        self.dropout = dropout

    def forward(self,
                x:          torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        x:           [N, F]
        edge_index:  [2, E]
        edge_weight: [E]  optional dynamic weights
        Returns:     [N, hidden_channels]
        """
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class DynamicDirectedSTGNN(nn.Module):
    """
    Full model:
      1. Per-timestep GCN  → spatial embedding  [lookback, N, hidden]
      2. Reshape           → [N, lookback, hidden]
      3. GRU               → temporal summary   [N, gru_hidden]
      4. Linear head       → targets            [N, num_targets]
    """

    def __init__(self,
                 in_channels:     int   = 11,
                 gcn_hidden:      int   = 64,
                 gcn_layers:      int   = 2,
                 gru_hidden:      int   = 128,
                 gru_layers:      int   = 2,
                 num_targets:     int   = 3,
                 dropout:         float = 0.3,
                 normalize:       bool  = True):
        super().__init__()

        self.gcn = TemporalGCN(
            in_channels=in_channels,
            hidden_channels=gcn_hidden,
            num_layers=gcn_layers,
            dropout=dropout,
            normalize=normalize,
        )

        self.gru = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_targets),
        )

    def forward(self,
                X:            torch.Tensor,
                edge_index:   torch.Tensor,
                edge_weight:  torch.Tensor = None) -> torch.Tensor:
        """
        X:           [batch, lookback, N, F]
        edge_index:  [2, E]
        edge_weight: [batch, E] or [E]  optional

        Returns:     [batch, N, num_targets]
        """
        batch, lookback, N, F = X.shape
        gcn_outs = []

        for t in range(lookback):
            x_t = X[:, t, :, :]                     # [batch, N, F]
            x_t = x_t.reshape(batch * N, F)          # merge batch×node for GCN

            # Offset edge_index for batched graph
            batch_edge_index = _batch_edge_index(edge_index, batch, N)

            ew = None
            if edge_weight is not None:
                ew = edge_weight if edge_weight.dim() == 1 \
                     else edge_weight[:, t % edge_weight.shape[1]
                                      if edge_weight.dim() > 1 else 0]
                ew = ew.repeat(batch) if ew.dim() == 1 else ew.reshape(-1)

            h = self.gcn(x_t, batch_edge_index, ew)  # [batch*N, gcn_hidden]
            h = h.reshape(batch, N, -1)               # [batch, N, gcn_hidden]
            gcn_outs.append(h)

        # Stack along time → [batch, N, lookback, gcn_hidden]
        spatial = torch.stack(gcn_outs, dim=2)

        # GRU expects [batch*N, lookback, gcn_hidden]
        B, Nn, T, H = spatial.shape
        spatial_flat = spatial.reshape(B * Nn, T, H)
        _, h_n = self.gru(spatial_flat)               # h_n: [gru_layers, B*N, gru_hidden]
        temporal = h_n[-1]                            # [B*N, gru_hidden]
        temporal = temporal.reshape(B, Nn, -1)        # [B, N, gru_hidden]

        out = self.head(temporal)                     # [B, N, num_targets]
        return out


def _batch_edge_index(edge_index: torch.Tensor,
                      batch_size: int,
                      n_nodes: int) -> torch.Tensor:
    """
    Replicate edge_index for a batched graph.
    Each graph in the batch has its node indices offset by n_nodes.
    """
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes
    offsets = offsets.unsqueeze(-1)                    # [batch, 1]
    ei_list = [edge_index + offsets[b] for b in range(batch_size)]
    return torch.cat(ei_list, dim=1)
