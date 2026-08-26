"""Causal-path reconstruction for ATHENA global interpretation (Proof §IV-E)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple, Union

# Event identity/order are retained so parallel events with the same endpoints
# and type remain distinct throughout interpretation.
PathEdge = Tuple[int, int, str, str, int]
AttackPath = List[PathEdge]


def _edge_order_key(graph, edge_id: int) -> Tuple[float, int, int]:
    attrs = graph.es[int(edge_id)].attributes()
    try:
        timestamp = float(attrs.get("timestamp", 0.0) or 0.0)
    except (TypeError, ValueError):
        timestamp = 0.0
    try:
        event_order = int(attrs.get("event_order", edge_id) or edge_id)
    except (TypeError, ValueError):
        event_order = int(edge_id)
    return timestamp, event_order, int(edge_id)


def _edge_triple(graph, edge_id: int) -> PathEdge:
    edge = graph.es[int(edge_id)]
    attrs = edge.attributes()
    return (
        int(edge.source),
        int(edge.target),
        str(attrs.get("actions", "")),
        str(attrs.get("event_id", edge_id)),
        int(attrs.get("event_order", edge_id)),
    )


def _path_from_edge_ids(graph, edge_ids: Sequence[int]) -> AttackPath:
    return [_edge_triple(graph, edge_id) for edge_id in edge_ids]


def _forward_causal_paths(
    graph,
    source: int,
    target: int,
    *,
    max_paths: int,
    max_expansions: int,
) -> Tuple[List[AttackPath], Dict]:
    """Enumerate forward, time-monotone simple paths within explicit budgets."""
    paths: List[AttackPath] = []
    stack = [(int(source), tuple(), frozenset({int(source)}), None)]
    expansions = 0
    truncated = False
    while stack:
        if len(paths) >= max_paths or expansions >= max_expansions:
            truncated = True
            break
        vertex, edge_ids, visited, previous_key = stack.pop()
        expansions += 1
        if vertex == int(target) and edge_ids:
            paths.append(_path_from_edge_ids(graph, edge_ids))
            continue
        incident = sorted(
            graph.incident(vertex, mode="OUT"),
            key=lambda edge_id: _edge_order_key(graph, edge_id),
            reverse=True,
        )
        for edge_id in incident:
            edge = graph.es[int(edge_id)]
            neighbour = int(edge.target)
            if neighbour in visited:
                continue
            edge_key = _edge_order_key(graph, edge_id)
            if previous_key is not None and edge_key < previous_key:
                continue
            stack.append((
                neighbour,
                edge_ids + (int(edge_id),),
                visited | {neighbour},
                edge_key,
            ))
    return paths, {"expansions": expansions, "truncated": truncated}


def _bounded_fallback_paths(
    graph,
    source: int,
    max_hops: int,
    *,
    max_paths: int,
    max_expansions: int,
) -> Tuple[List[AttackPath], Dict]:
    """Enumerate bounded forward causal paths around an isolated alert."""
    paths: List[AttackPath] = []
    stack = [(source, tuple(), frozenset({source}), None)]
    expansions = 0
    truncated = False
    while stack:
        if len(paths) >= max_paths or expansions >= max_expansions:
            truncated = True
            break
        vertex, edge_ids, visited, previous_key = stack.pop()
        expansions += 1
        if len(edge_ids) >= max_hops:
            if edge_ids:
                paths.append(_path_from_edge_ids(graph, edge_ids))
            continue
        incident = sorted(
            graph.incident(vertex, mode="OUT"),
            key=lambda edge_id: _edge_order_key(graph, edge_id),
            reverse=True,
        )
        extended = False
        for edge_id in incident:
            edge = graph.es[int(edge_id)]
            neighbour = int(edge.target)
            if neighbour in visited:
                continue
            edge_key = _edge_order_key(graph, edge_id)
            if previous_key is not None and edge_key < previous_key:
                continue
            extended = True
            stack.append((
                neighbour,
                edge_ids + (int(edge_id),),
                visited | {neighbour},
                edge_key,
            ))
        if edge_ids and not extended:
            paths.append(_path_from_edge_ids(graph, edge_ids))
    return paths, {"expansions": expansions, "truncated": truncated}


def reconstruct_attack_paths(
    graph,
    source: int,
    malicious_nodes: Iterable[int],
    fallback_hops: int = 4,
    *,
    max_paths_per_peer: int = 1024,
    max_paths_per_alert: int = 4096,
    max_expansions_per_alert: int = 100_000,
    return_audit: bool = False,
) -> Union[List[AttackPath], Tuple[List[AttackPath], Dict]]:
    """Build one malicious node's causal path set as specified in Proof §IV-E.

    Directed paths connect the node to every reachable malicious peer.  The
    bounded lambda-hop fallback is used only when no such peer path exists.
    """
    limits = {
        "max_paths_per_peer": max(1, int(max_paths_per_peer)),
        "max_paths_per_alert": max(1, int(max_paths_per_alert)),
        "max_expansions_per_alert": max(1, int(max_expansions_per_alert)),
    }
    if graph is None or not 0 <= int(source) < graph.vcount():
        audit = {**limits, "source": int(source), "path_count": 0, "expansions": 0,
                 "fallback_used": False, "truncated": False, "peer_audit": []}
        return ([], audit) if return_audit else []
    source = int(source)
    peers = sorted({
        int(v) for v in malicious_nodes
        if int(v) != source and 0 <= int(v) < graph.vcount()
    })
    reachable_vertices = set(int(value) for value in graph.subcomponent(source, mode="OUT"))
    reachable_peers = [peer for peer in peers if peer in reachable_vertices]
    skipped_unreachable = [peer for peer in peers if peer not in reachable_vertices]
    paths: List[AttackPath] = []
    peer_audit = []
    expansions = 0
    truncated = False
    skipped_budget = []
    peer_count = len(reachable_peers)
    for peer_index, peer in enumerate(reachable_peers):
        remaining_paths = limits["max_paths_per_alert"] - len(paths)
        remaining_expansions = limits["max_expansions_per_alert"] - expansions
        # Allocate the total budget fairly before traversal so an early peer
        # (especially one with no time-monotone path) cannot starve later peers.
        path_quota = (
            limits["max_paths_per_alert"] // peer_count
            + int(peer_index < limits["max_paths_per_alert"] % peer_count)
        ) if peer_count else 0
        expansion_quota = (
            limits["max_expansions_per_alert"] // peer_count
            + int(peer_index < limits["max_expansions_per_alert"] % peer_count)
        ) if peer_count else 0
        if remaining_paths <= 0 or remaining_expansions <= 0 or path_quota <= 0 or expansion_quota <= 0:
            truncated = True
            skipped_budget.append(peer)
            continue
        peer_paths, peer_meta = _forward_causal_paths(
            graph,
            source,
            peer,
            max_paths=min(limits["max_paths_per_peer"], remaining_paths, path_quota),
            max_expansions=min(remaining_expansions, expansion_quota),
        )
        paths.extend(peer_paths)
        expansions += int(peer_meta["expansions"])
        truncated = truncated or bool(peer_meta["truncated"])
        peer_audit.append({"peer": peer, "path_count": len(peer_paths), **peer_meta})
    fallback_used = not paths
    if not paths:
        remaining_expansions = limits["max_expansions_per_alert"] - expansions
        if remaining_expansions > 0:
            paths, fallback_meta = _bounded_fallback_paths(
                graph,
                source,
                max(1, int(fallback_hops)),
                max_paths=limits["max_paths_per_alert"],
                max_expansions=remaining_expansions,
            )
            expansions += int(fallback_meta["expansions"])
            truncated = truncated or bool(fallback_meta["truncated"])
        else:
            truncated = True

    unique: List[AttackPath] = []
    seen = set()
    for path in paths:
        signature = tuple(path)
        if signature not in seen:
            seen.add(signature)
            unique.append(path)
    audit = {
        **limits,
        "source": source,
        "peer_count": len(peers),
        "reachable_peer_count": len(reachable_peers),
        "processed_peers": [row["peer"] for row in peer_audit],
        "skipped_unreachable_peers": skipped_unreachable,
        "skipped_budget_peers": skipped_budget,
        "path_count": len(unique),
        "expansions": expansions,
        "fallback_used": fallback_used,
        "truncated": truncated,
        "peer_audit": peer_audit,
    }
    return (unique, audit) if return_audit else unique


__all__ = ["PathEdge", "AttackPath", "reconstruct_attack_paths"]
