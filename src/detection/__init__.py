"""Paper §IV.D - Adaptive contrastive learning components.

Public surface of the detection module:

- :class:`GINEncoder`, :class:`GINConv`, :class:`MLP` - 3-layer GIN.
- :class:`TemporalNodeEncoder` - final-GIN GRU that updates each entity once
  per time-windowed snapshot.
- :class:`ATHENAEncoder` - orchestrates GIN + GRU inside the
  hard-sample-weighted supervised contrastive training loop.
- :class:`ATHENADetector` - two-layer MLP head producing per-snapshot binary
  anomaly scores.
"""
from .gin_encoder import (
    GINEncoder,
    GINConv,
    MLP,
)
from .temporal_encoder import TemporalNodeEncoder
from .contrastive_learning import ATHENAEncoder
from .classifier import MLPClassify as ATHENADetector

__all__ = [
    "GINEncoder",
    "GINConv",
    "MLP",
    "TemporalNodeEncoder",
    "ATHENAEncoder",
    "ATHENADetector",
]
