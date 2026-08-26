"""Build the node-centred r-hop graph units defined in paper Section IV-B."""
from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from collections.abc import Sequence
from typing import Iterable


@dataclass(frozen=True)
class GraphUnitRef:
    snapshot_id: int
    anchor_node: int
    anchor_name: str

    @property
    def key(self) -> tuple[int, int]:
        return self.snapshot_id, self.anchor_node


class LazyGraphUnits(Sequence):
    """Reference-first r-hop units with a bounded materialization cache."""

    def __init__(self, snapshots, refs, r_hop: int, max_cached_units: int = 32):
        self.snapshots = snapshots
        self.refs = list(refs)
        self.r_hop = int(r_hop)
        self.max_cached_units = max(0, int(max_cached_units))
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.refs)

    def iter_refs(self):
        return iter(self.refs)

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def materialize(self, ref: GraphUnitRef):
        key = ref.key
        if key in self._cache:
            unit = self._cache.pop(key)
            self._cache[key] = unit
            return unit
        graph = self.snapshots[ref.snapshot_id]
        vertices = sorted(set(graph.neighborhood(ref.anchor_node, order=self.r_hop, mode="all")))
        unit = graph.induced_subgraph(vertices)
        unit.vs["_athena_anchor"] = [False] * unit.vcount()
        unit.vs[vertices.index(ref.anchor_node)]["_athena_anchor"] = True
        if self.max_cached_units:
            self._cache[key] = unit
            while len(self._cache) > self.max_cached_units:
                self._cache.popitem(last=False)
        return unit

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [(self.materialize(ref), ref) for ref in self.refs[index]]
        ref = self.refs[index]
        return self.materialize(ref), ref

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return list(self) == list(other)
        return NotImplemented


def build_graph_units(
    handler, snapshot_ids: Iterable[int], r_hop: int, *, max_cached_units: int = 32,
):
    """Return ``(benign_units, attack_units)`` from training snapshots only.

    Each unit is ``(induced_r_hop_graph, GraphUnitRef)``.  The anchor label
    determines whether the unit belongs to the benign or attack corpus.
    """
    benign_refs = []
    attack_refs = []
    snapshots = getattr(handler, "snapshots", [])
    for snapshot_id in sorted({int(value) for value in snapshot_ids}):
        if snapshot_id < 0 or snapshot_id >= len(snapshots):
            continue
        graph = snapshots[snapshot_id]
        if graph is None or graph.vcount() == 0:
            continue
        labels = graph.vs["label"] if "label" in graph.vs.attributes() else [0] * graph.vcount()
        for anchor_node in range(graph.vcount()):
            attrs = graph.vs[anchor_node].attributes()
            anchor_name = str(attrs.get("name", attrs.get("uuid", anchor_node)))
            ref = GraphUnitRef(snapshot_id, anchor_node, anchor_name)
            if int(labels[anchor_node] or 0) == 1:
                attack_refs.append(ref)
            else:
                benign_refs.append(ref)
    return (
        LazyGraphUnits(snapshots, benign_refs, r_hop, max_cached_units),
        LazyGraphUnits(snapshots, attack_refs, r_hop, max_cached_units),
    )
