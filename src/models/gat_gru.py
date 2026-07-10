"""
gat_gru.py
==========
Phase 2 Architecture: GAT (Graph Attention Network) + GRU
Switch to this after the GCN+GRU baseline is established.

Key difference: attention coefficients replace fixed GCN aggregation,
allowing the model to learn which neighbouring sub-basins matter more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm


class GATTemporalBlock(nn.Module):
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 heads:        int   = 4,
                 dropout:      float = 0.3,
                 num_layers:   int   = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        self.convs.append(
            GATv2Conv(in_channels, out_channels // heads,
                      heads=heads, dropout=dropout, edge_dim=1)
        )
        self.bns.append(BatchNorm(out_channels))

        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(out_channels, out_channels // heads,
                          heads=heads, dropout=dropout, edge_dim=1)
            )
            self.bns.append(BatchNorm(out_channels))

        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATGRUModel(nn.Module):
    """GAT + GRU — Phase 2 model."""

    def __init__(self,
                 in_channels:  int   = 11,
                 gat_hidden:   int   = 64,
                 gat_heads:    int   = 4,
                 gat_layers:   int   = 2,
                 gru_hidden:   int   = 128,
                 gru_layers:   int   = 2,
                 num_targets:  int   = 3,
                 dropout:      float = 0.3):
        super().__init__()

        self.gat = GATTemporalBlock(
            in_channels=in_channels,
            out_channels=gat_hidden,
            heads=gat_heads,
            dropout=dropout,
            num_layers=gat_layers,
        )

        self.gru = nn.GRU(
            input_size=gat_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, num_targets),
        )

    def forward(self, X, edge_index, edge_attr=None, edge_weight=None):
        batch, lookback, N, F = X.shape
        gcn_outs = []

        for t in range(lookback):
            x_t   = X[:, t].reshape(batch * N, F)
            batch_ei = _batch_edge_index(edge_index, batch, N)
            ea = None
            if edge_weight is not None:
                ew = edge_weight.reshape(-1, 1) if edge_weight.dim() == 1 \
                     else edge_weight[:, t % edge_weight.shape[1]].reshape(-1, 1)
                ea = ew
            h = self.gat(x_t, batch_ei, ea)
            gcn_outs.append(h.reshape(batch, N, -1))

        spatial = torch.stack(gcn_outs, dim=2)         # [B, N, T, H]
        B, Nn, T, H = spatial.shape
        flat = spatial.reshape(B * Nn, T, H)
        _, h_n = self.gru(flat)
        temporal = h_n[-1].reshape(B, Nn, -1)
        return self.head(temporal)


def _batch_edge_index(edge_index, batch_size, n_nodes):
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes
    return torch.cat([edge_index + offsets[b] for b in range(batch_size)], dim=1)
