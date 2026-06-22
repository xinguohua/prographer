"""Paper §IV.D - Adaptive contrastive learning components.

Public surface of the detection module:

- :class:`GINEncoder`, :class:`TypedGINConv`, :class:`MLP` - typed 3-layer GIN.
- :class:`TemporalPerLayer` - per-layer GRU that propagates node state across
  time-windowed snapshots.
- :class:`ATHENAEncoder` - orchestrates GIN + GRU + StrategyMoE inside the
  hard-sample-weighted supervised contrastive training loop.
- :class:`ATHENADetector` - two-layer MLP head producing per-snapshot binary
  anomaly scores.
"""
from .gin_encoder import (
    GINEncoder,
    TypedGINConv,
    MLP,
    EDGE_CATEGORY,
    NUM_EDGE_CATEGORIES,
    classify_edge,
)
from .temporal_encoder import TemporalPerLayer
from .contrastive_learning import GCCEmbedderDev as ATHENAEncoder, StrategyMoE
from .classifier import MLPClassify as ATHENADetector

__all__ = [
    "GINEncoder",
    "TypedGINConv",
    "MLP",
    "EDGE_CATEGORY",
    "NUM_EDGE_CATEGORIES",
    "classify_edge",
    "TemporalPerLayer",
    "ATHENAEncoder",
    "StrategyMoE",
    "ATHENADetector",
]
