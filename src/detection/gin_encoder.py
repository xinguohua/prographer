"""Paper §IV-D three-layer Graph Isomorphism Network encoder."""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GINConv(nn.Module):
    """Sum-aggregating GIN layer with a learnable self weight."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = MLP(in_dim, out_dim, out_dim, dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                **kwargs) -> torch.Tensor:
        n = x.size(0)
        aggregate = torch.zeros_like(x)
        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            aggregate.index_add_(0, dst, x[src])
        return self.mlp((1.0 + self.eps) * x + aggregate)


class GINEncoder(nn.Module):
    """Stack of GIN layers. Default is the 3-layer GIN reported in the
    paper. Returns either the final layer's hidden states or the list of all
    layer states when ``return_all=True``."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()
        num_layers = int(max(1, num_layers))
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GINConv(dims[i], dims[i + 1], dropout=dropout) for i in range(num_layers)
        ])
        self.layer_dims = dims[1:]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor = None, return_all: bool = False, **kwargs):
        del edge_feat  # Proof equations (2)-(3) aggregate neighboring node states.
        Zs: List[torch.Tensor] = []
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            Zs.append(h)
        return Zs if return_all else h
