"""Snapshot-level temporal node encoder from paper Eq. (4).

The GIN first produces one instantaneous representation per node for the
current snapshot. A single GRU cell fuses that final-layer vector with the
same entity's previous state. State is committed once per entity and snapshot.
"""
from __future__ import annotations
from typing import Dict, List, Mapping, Optional

import torch
import torch.nn as nn


class TemporalNodeEncoder(nn.Module):
    """Name-keyed GRU state over final-layer GIN representations."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.cell = nn.GRUCell(self.embedding_dim, self.embedding_dim)
        self.table: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.table.clear()

    def fetch(
        self,
        node_ids: List[str],
        device: torch.device,
        table: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Fetch state from the live table or an immutable time-point copy."""
        source = self.table if table is None else table
        previous = torch.zeros(
            (len(node_ids), self.embedding_dim),
            dtype=torch.float32,
            device=device,
        )
        for index, node_id in enumerate(node_ids):
            stored = source.get(str(node_id))
            if stored is not None:
                previous[index] = stored.to(device)
        return previous

    def forward(self, instantaneous: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        return self.cell(instantaneous, previous)

    def commit(self, node_ids: List[str], hidden: torch.Tensor) -> None:
        if hidden.size(0) != len(node_ids):
            raise ValueError("node_ids and hidden state have different lengths")
        for index, node_id in enumerate(node_ids):
            self.table[str(node_id)] = hidden[index].detach().cpu().contiguous()

    def snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a detached copy for same-time alternative graph encodings."""
        return {key: value.detach().clone() for key, value in self.table.items()}
