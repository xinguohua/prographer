"""Paper §IV.E — Key causal path extraction from flagged snapshots.

For each snapshot flagged as malicious by the detector, extract the key
causal sub-path that connects the malicious node(s) to other entities. The
extraction walks the provenance edges in reverse-time order starting from
each malicious node and follows the chain of actions that materially
contributed to the malicious behaviour.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Set, Tuple

try:
    import igraph as ig  # noqa: F401
except ImportError:
    ig = None


def extract_attack_subgraph(
    g,
    malicious_nodes: Iterable[int],
    max_hops: int = 4,
    max_nodes: int = 64,
):
    """Extract the key attack subgraph rooted at the given malicious nodes.

    The expansion is BFS-bounded by ``max_hops`` and ``max_nodes``; nodes are
    visited in causal order (incoming edges first to capture the upstream
    cause chain, outgoing edges next for downstream effects).
    """
    visited: Set[int] = set()
    queue: deque = deque()
    for v in malicious_nodes:
        if 0 <= v < g.vcount():
            visited.add(v)
            queue.append((v, 0))

    while queue and len(visited) < max_nodes:
        v, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for u in g.neighbors(v, mode="in"):
            if u not in visited:
                visited.add(u)
                queue.append((u, depth + 1))
                if len(visited) >= max_nodes:
                    break
        for u in g.neighbors(v, mode="out"):
            if u not in visited:
                visited.add(u)
                queue.append((u, depth + 1))
                if len(visited) >= max_nodes:
                    break

    return g.subgraph(sorted(visited))


def extract_key_path(g, malicious_nodes: Iterable[int]) -> List[Tuple[int, int, str]]:
    """Return a list of ``(source, target, action)`` triples ordered by edge
    timestamp, restricted to edges that touch any malicious node or any
    upstream ancestor reachable through type-coherent process edges."""
    if g is None:
        return []
    malicious_set: Set[int] = {int(v) for v in malicious_nodes if 0 <= int(v) < g.vcount()}
    if not malicious_set:
        return []

    closure = set(malicious_set)
    frontier = deque(malicious_set)
    while frontier:
        v = frontier.popleft()
        for u in g.neighbors(v, mode="in"):
            if u in closure:
                continue
            t = str(g.vs[u].attributes().get("type", "")).lower()
            if "process" in t or "subject" in t:
                closure.add(u)
                frontier.append(u)

    edges: List[Tuple[int, int, str]] = []
    timestamps: List[float] = []
    for e in g.es:
        if e.source in closure or e.target in closure:
            ts_raw = e.attributes().get("timestamp", 0)
            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                ts = 0.0
            edges.append((e.source, e.target, str(e.attributes().get("actions", ""))))
            timestamps.append(ts)

    order = sorted(range(len(edges)), key=lambda i: timestamps[i])
    return [edges[i] for i in order]
