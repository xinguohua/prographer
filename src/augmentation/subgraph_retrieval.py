"""Weisfeiler--Lehman subtree-kernel retrieval (Proof Eq. (1)).

Initial node labels combine entity type with snapshot-local semantic
attributes; edge labels retain the audited operation.  The normalized kernel
is used to rank attack graph units for a benign anchor.
"""
from __future__ import annotations
from collections import Counter
from typing import List, Optional
import hashlib
import math

from .property_adapter import property_values, semantic_summary, typed_fields

try:
    import igraph as ig
except ImportError:
    ig = None


def _node_initial_label(g, v_idx: int) -> str:
    """Return an entity-type/attribute label without using the UUID name."""
    attrs = g. vs [v_idx].attributes()
    entity_type = str(attrs.get("type", "UNK")).lower()
    raw_prop = attrs.get("properties", "")

    # Derive a coarse semantic label from the snapshot-local properties.
    if "process" in entity_type or "subject" in entity_type:
        coarse = semantic_summary(raw_prop, entity_type, max_chars=128)
    elif "file" in entity_type:
        # Retain a coarse path prefix for file entities.
        fields = typed_fields(raw_prop, entity_type)
        paths = sorted(fields.get("properties.path", set()))
        fp = paths[0] if paths else "UNK"
        segments = fp.split("/")[:4]
        coarse = "/".join(segments) if segments and segments[0] != "" else fp[:24]
    elif "net" in entity_type or "flow" in entity_type or "sock" in entity_type:
        # Retain one normalized address for network entities.
        fields = typed_fields(raw_prop, entity_type)
        addresses = sorted(fields.get("properties.address", set()))
        coarse = f"address:{addresses[0]}" if addresses else "net"
    else:
        values = property_values(raw_prop)
        coarse = values[0][:12] if values else "UNK"

    return f"{entity_type}|{coarse}"


def _edge_label(g, e_idx: int) -> str:
    """Return the audited operation used as the WL edge label."""
    attrs = g.es[e_idx].attributes()
    return str(attrs.get("actions", "UNK"))


def wl_subtree_labels(g, h: int = 3) -> List[Counter]:
    """Run ``h`` WL iterations and return one label histogram per round.

    Args:
        g: ``igraph.Graph`` to encode.
        h: Number of WL refinement iterations.

    Returns:
        A list of ``h + 1`` label-frequency counters, including round zero.
    """
    n = g.vcount()
    if n == 0:
        return [Counter() for _ in range(h + 1)]

    labels = [_node_initial_label(g, i) for i in range(n)]
    histograms = [Counter(labels)]

    for _ in range(h):
        new_labels = []
        for v in range(n):
            neighbor_labels = []
            for e_idx in g.incident(v, mode="all"):
                e = g.es[e_idx]
                src, dst = e.source, e.target
                neighbor = dst if src == v else src
                edge_lbl = _edge_label(g, e_idx)
                neighbor_labels.append(f"{edge_lbl}:{labels[neighbor]}")
            neighbor_labels.sort()
            combined = labels[v] + "|" + "|".join(neighbor_labels)
            new_labels.append(hashlib.md5(combined.encode()).hexdigest()[:16])
        labels = new_labels
        histograms.append(Counter(labels))

    return histograms


def wl_kernel(g1, g2, h: int = 3) -> float:
    """Return the normalized WL-subtree inner product for two graphs.

    K_WL(g1, g2) = sum_i <phi_i(g1), phi_i(g2)> / (||phi(g1)|| * ||phi(g2)||)

    Args:
        g1, g2: Graphs to compare.
        h: Number of WL refinement iterations.

    Returns:
        Normalized similarity in ``[0, 1]``.
    """
    hist1 = wl_subtree_labels(g1, h)
    hist2 = wl_subtree_labels(g2, h)

    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for c1, c2 in zip(hist1, hist2):
        all_keys = set(c1.keys()) | set(c2.keys())
        for k in all_keys:
            v1 = c1.get(k, 0)
            v2 = c2.get(k, 0)
            dot += v1 * v2
            norm1_sq += v1 * v1
            norm2_sq += v2 * v2

    norm1 = math.sqrt(norm1_sq) if norm1_sq > 0 else 1e-12
    norm2 = math.sqrt(norm2_sq) if norm2_sq > 0 else 1e-12

    return dot / (norm1 * norm2)


def _kernel_from_histograms(hist1, hist2) -> float:
    """Compute the normalized kernel from two WL histogram sequences."""
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    for c1, c2 in zip(hist1, hist2):
        all_keys = set(c1.keys()) | set(c2.keys())
        for k in all_keys:
            v1 = c1.get(k, 0)
            v2 = c2.get(k, 0)
            dot += v1 * v2
            norm1_sq += v1 * v1
            norm2_sq += v2 * v2
    norm1 = math.sqrt(norm1_sq) if norm1_sq > 0 else 1e-12
    norm2 = math.sqrt(norm2_sq) if norm2_sq > 0 else 1e-12
    return dot / (norm1 * norm2)


def top_k_similar_attacks(benign_graph, attack_graphs: list, k: int = 5, h: int = 3,
                          _attack_hist_cache: dict = None) -> list:
    """Retrieve the top-``k`` attack units for a benign anchor (Proof Eq. (1)).

    Args:
        benign_graph: Benign anchor graph unit.
        attack_graphs: Eager ``(graph, reference)`` pairs or a lazy graph-unit
            collection exposing ``iter_refs`` and ``materialize``.
        k: Maximum number of ranked references to return.
        h: Number of WL refinement iterations.
        _attack_hist_cache: Optional histogram cache keyed by reference.

    Returns:
        ``(graph, reference, similarity)`` tuples in descending score order.
    """
    hist_b = wl_subtree_labels(benign_graph, h)

    scored = []
    lazy = hasattr(attack_graphs, "iter_refs") and hasattr(attack_graphs, "materialize")
    source = ((None, ref) for ref in attack_graphs.iter_refs()) if lazy else iter(attack_graphs)
    for ag, reference in source:
        cache_key = getattr(reference, "key", reference)
        # Reuse cached reference histograms when available.
        if _attack_hist_cache is not None and cache_key in _attack_hist_cache:
            hist_a = _attack_hist_cache[cache_key]
        else:
            if ag is None:
                ag = attack_graphs.materialize(reference)
            hist_a = wl_subtree_labels(ag, h)
            if _attack_hist_cache is not None:
                _attack_hist_cache[cache_key] = hist_a
        sim = _kernel_from_histograms(hist_b, hist_a)
        scored.append((None if lazy else ag, reference, sim))

    scored.sort(key=lambda x: -x[2])
    selected = scored[:k]
    if lazy:
        selected = [
            (ag if ag is not None else attack_graphs.materialize(reference), reference, sim)
            for ag, reference, sim in selected
        ]
    return selected
