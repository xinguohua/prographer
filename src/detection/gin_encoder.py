"""Paper §IV.D - 3-layer Typed GIN encoder.

Group-aggregated GIN over four edge categories (process, file, network, memory).
Aggregating per category prevents high-volume read/write edges from drowning
out execute/fork signal during message passing.
"""
from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# Four semantic edge groups: process ops, file ops, network ops, memory ops.
EDGE_CATEGORY = {
    'EVENT_EXECUTE': 0, 'EVENT_FORK': 0, 'EVENT_CLONE': 0, 'EVENT_EXIT': 0,
    'EVENT_READ': 1, 'EVENT_WRITE': 1, 'EVENT_OPEN': 1, 'EVENT_CLOSE': 1,
    'EVENT_UNLINK': 1, 'EVENT_RENAME': 1,
    'EVENT_CONNECT': 2, 'EVENT_SENDTO': 2, 'EVENT_RECVFROM': 2, 'EVENT_ACCEPT': 2,
    'EVENT_MMAP': 3, 'EVENT_MPROTECT': 3,
}
NUM_EDGE_CATEGORIES = 4


def classify_edge(action_str: str) -> int:
    """Map an edge action string to a category index 0..3. Multi-action edges
    take the first recognised verb; unknown actions fall back to file ops."""
    for act in action_str.split(','):
        act = act.strip()
        if act in EDGE_CATEGORY:
            return EDGE_CATEGORY[act]
    return 1


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TypedGINConv(nn.Module):
    """Group-aggregating GIN convolution.

    Neighbours are split by edge category and aggregated independently, then
    concatenated with self features and passed through an MLP. Execute-class
    neighbours stay in their own channel and are not diluted by read/write
    neighbours.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_categories: int = NUM_EDGE_CATEGORIES,
                 dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(in_dim * (num_categories + 1), out_dim, out_dim, dropout)
        self.num_categories = num_categories

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_cat: torch.Tensor = None, **kwargs) -> torch.Tensor:
        n = x.size(0)
        d = x.size(1)

        if edge_index.numel() == 0 or edge_cat is None:
            agg = torch.zeros(n, d * self.num_categories, device=x.device)
        else:
            src, dst = edge_index[0], edge_index[1]
            agg_parts = []
            for cat_id in range(self.num_categories):
                mask = (edge_cat == cat_id)
                cat_agg = torch.zeros(n, d, device=x.device)
                if mask.any():
                    cat_agg.index_add_(0, dst[mask], x[src[mask]])
                agg_parts.append(cat_agg)
            agg = torch.cat(agg_parts, dim=1)

        combined = torch.cat([x, agg], dim=1)
        return self.mlp(combined)


class GINEncoder(nn.Module):
    """Stack of TypedGINConv layers. Default is the 3-layer GIN reported in the
    paper. Returns either the final layer's hidden states or the list of all
    layer states when ``return_all=True``."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, dropout: float = 0.1, **kwargs):
        super().__init__()
        num_layers = int(max(1, num_layers))
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            TypedGINConv(dims[i], dims[i + 1], dropout=dropout) for i in range(num_layers)
        ])
        self.layer_dims = dims[1:]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_feat: torch.Tensor = None, return_all: bool = False, **kwargs):
        Zs: List[torch.Tensor] = []
        h = x
        for conv in self.layers:
            h = conv(h, edge_index, edge_cat=edge_feat)
            h = F.relu(h)
            Zs.append(h)
        return Zs if return_all else h
