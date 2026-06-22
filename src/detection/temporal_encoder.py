"""Paper §IV.D - Per-layer GRU temporal encoder.

Maintains one GRUCell per GIN layer and a name-keyed hidden-state table so
that a node's representation can persist across time-windowed snapshots even
when the node-set changes between windows.
"""
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn


class TemporalPerLayer(nn.Module):
    def __init__(self, layer_dims: List[int]):
        super().__init__()
        self.layer_dims = [int(d) for d in layer_dims]
        self.cells = nn.ModuleList([nn.GRUCell(d, d) for d in self.layer_dims])
        # One name-keyed hidden state table per layer.
        self.tables: List[Dict[str, torch.Tensor]] = [dict() for _ in self.layer_dims]

    def reset(self):
        for t in self.tables:
            t.clear()

    def fetch(self, node_ids: List[str], device: torch.device) -> List[torch.Tensor]:
        """Look up previous hidden states for ``node_ids`` at every layer;
        nodes not yet seen get a zero state."""
        H_prev: List[torch.Tensor] = []
        n = len(node_ids)
        for li, dim in enumerate(self.layer_dims):
            table = self.tables[li]
            H = torch.zeros((n, dim), dtype=torch.float32, device=device)
            for i, nid in enumerate(node_ids):
                if nid in table:
                    H[i] = table[nid].to(device)
            H_prev.append(H)
        return H_prev

    def forward(self, Z_list: List[torch.Tensor], H_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        H_list: List[torch.Tensor] = []
        for li, cell in enumerate(self.cells):
            h1 = cell(Z_list[li], H_prev[li])
            H_list.append(h1)
        return H_list

    def commit(self, node_ids: List[str], H_list: List[torch.Tensor]):
        """Persist the new hidden states for ``node_ids`` into the per-layer tables."""
        for li in range(min(len(self.tables), len(H_list))):
            table = self.tables[li]
            Hl = H_list[li]
            for i, nid in enumerate(node_ids):
                table[nid] = Hl[i].detach().to('cpu').contiguous()
