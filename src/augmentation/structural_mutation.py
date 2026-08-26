"""Paper §IV.B - Structural mutation (Algorithm 1).

For each benign anchor G_b:
1. WL-subtree retrieval yields the Top-K most similar attack snapshots.
2. For each attack snapshot, score every compatible cross-graph node pair
   with Jaccard similarity, then BFS-align an attack region S' from the best
   seeds.
3. Replace the aligned region of G_b with S' to produce the mutated graph.
"""
from __future__ import annotations
import re
from collections import deque
from typing import List, Tuple, Dict, Optional

try:
    import igraph as ig
except ImportError:
    ig = None


def _token_set(g, v_idx: int) -> set:
    """Return the semantic node-label tokens used by Jaccard alignment."""
    attrs = g. vs [v_idx].attributes()
    tokens = set()
    for k, v in attrs.items():
        key = str(k).lower()
        if key in {"label", "name", "uuid", "timestamp"} or key.startswith("_athena_"):
            continue
        values = re.findall(r"[a-z0-9_./:-]+", str(v).lower())
        tokens.update(f"{key}:{value}" for value in values)
    return tokens


def _jaccard_tok(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union > 0 else 0.0


def _causal_edge_key(graph, edge_index: int) -> tuple:
    attrs = graph.es[int(edge_index)].attributes()
    try:
        timestamp = float(attrs.get("timestamp", 0.0) or 0.0)
    except (TypeError, ValueError):
        timestamp = 0.0
    try:
        event_order = int(attrs.get("event_order", edge_index) or edge_index)
    except (TypeError, ValueError):
        event_order = int(edge_index)
    return timestamp, event_order, int(edge_index)


def _first_causal_path(graph, source: int, target: int, allowed: set[int]) -> Optional[list[int]]:
    """Return the deterministic shortest forward time-monotone event path."""
    queue = deque([(int(source), tuple(), frozenset({int(source)}), None)])
    while queue:
        vertex, edge_path, visited, previous_key = queue.popleft()
        if vertex == int(target) and edge_path:
            return list(edge_path)
        outgoing = sorted(
            graph.incident(vertex, mode="OUT"),
            key=lambda edge_index: _causal_edge_key(graph, edge_index),
        )
        for edge_index in outgoing:
            edge = graph.es[int(edge_index)]
            neighbour = int(edge.target)
            edge_key = _causal_edge_key(graph, edge_index)
            if neighbour not in allowed or neighbour in visited:
                continue
            if previous_key is not None and edge_key < previous_key:
                continue
            queue.append((
                neighbour,
                edge_path + (int(edge_index),),
                visited | {neighbour},
                edge_key,
            ))
    return None


def _attack_chain_spec(graph, attack_region: set[int]) -> Optional[dict]:
    """Define the anchor-to-malicious causal events that S' must preserve."""
    anchors = [
        index for index in range(graph.vcount())
        if bool(graph.vs[index].attributes().get("_athena_anchor", False))
        and int(graph.vs[index].attributes().get("label", 0) or 0) == 1
    ]
    if not anchors or any(anchor not in attack_region for anchor in anchors):
        return None
    malicious = [
        index for index in range(graph.vcount())
        if int(graph.vs[index].attributes().get("label", 0) or 0) == 1
    ]
    required_nodes = set(anchors)
    required_edges = []
    seen_edges = set()
    all_nodes = set(range(graph.vcount()))
    for anchor in anchors:
        for peer in malicious:
            if peer == anchor:
                continue
            full_path = _first_causal_path(graph, anchor, peer, all_nodes)
            if full_path is None:
                continue
            region_path = _first_causal_path(graph, anchor, peer, attack_region)
            if region_path is None:
                return None
            for edge_index in region_path:
                edge = graph.es[int(edge_index)]
                attrs = edge.attributes()
                event_id = str(attrs.get("event_id", edge_index))
                key = (int(edge.source), int(edge.target), event_id)
                required_nodes.update((int(edge.source), int(edge.target)))
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                required_edges.append({
                    "source_attack_vertex": int(edge.source),
                    "target_attack_vertex": int(edge.target),
                    "event_id": event_id,
                    "event_order": int(
                        edge_index if attrs.get("event_order") is None else attrs.get("event_order")
                    ),
                    "action": str(attrs.get("actions", "")),
                })
    return {
        "required_attack_vertices": sorted(required_nodes),
        "required_attack_edges": required_edges,
    }


def aligned_region_search(g_b, g_a, r_hop: int = 4) -> Optional[Tuple[list, list, dict, float]]:
    """Return the paper-defined best alignment for one retrieved reference.

    The highest-Jaccard compatible node pair seeds a greedy alignment.  Each
    step adds the highest-Jaccard unmatched neighbour pair of any matched pair,
    bounded by ``r_hop`` around the benign seed.  The score is the mean Jaccard
    similarity of all matched pairs.
    """
    if g_b.vcount() == 0 or g_a.vcount() == 0:
        return None
    benign_tokens = {v: _token_set(g_b, v) for v in range(g_b.vcount())}
    attack_tokens = {w: _token_set(g_a, w) for w in range(g_a.vcount())}
    seed_pairs = []
    for w in range(g_a.vcount()):
        attack_type = str(g_a.vs[w].attributes().get("type", "")).lower()
        for v in range(g_b.vcount()):
            benign_type = str(g_b.vs[v].attributes().get("type", "")).lower()
            if attack_type != benign_type:
                continue
            seed_pairs.append((_jaccard_tok(attack_tokens[w], benign_tokens[v]), w, v))
    if not seed_pairs:
        return None
    seed_similarity, attack_seed, benign_seed = max(seed_pairs, key=lambda item: (item[0], -item[1], -item[2]))
    allowed_benign = set(g_b.neighborhood(benign_seed, order=int(r_hop), mode="all"))
    allowed_attack = set(g_a.neighborhood(attack_seed, order=int(r_hop), mode="all"))
    mapping = {attack_seed: benign_seed}
    attack_region = {attack_seed}
    benign_region = {benign_seed}
    pair_similarities = [seed_similarity]

    while True:
        candidates = []
        for matched_attack, matched_benign in mapping.items():
            attack_neighbors = (
                set(g_a.neighbors(matched_attack, mode="all"))
                & allowed_attack
                - attack_region
            )
            benign_neighbors = (
                set(g_b.neighbors(matched_benign, mode="all"))
                & allowed_benign
                - benign_region
            )
            for attack_node in attack_neighbors:
                attack_type = str(g_a.vs[attack_node].attributes().get("type", "")).lower()
                for benign_node in benign_neighbors:
                    benign_type = str(g_b.vs[benign_node].attributes().get("type", "")).lower()
                    if attack_type != benign_type:
                        continue
                    similarity = _jaccard_tok(attack_tokens[attack_node], benign_tokens[benign_node])
                    candidates.append((similarity, attack_node, benign_node))
        if not candidates:
            break
        similarity, attack_node, benign_node = max(
            candidates, key=lambda item: (item[0], -item[1], -item[2])
        )
        mapping[attack_node] = benign_node
        attack_region.add(attack_node)
        benign_region.add(benign_node)
        pair_similarities.append(similarity)

    if len(mapping) < 2:
        return None
    mean_similarity = sum(pair_similarities) / len(pair_similarities)
    return list(benign_region), list(attack_region), mapping, mean_similarity


def subgraph_replacement(
    g_b, g_a,
    S_b_nodes: list,
    S_a_nodes: list,
    pi: dict,
    r_hop: int = 4,
) -> Optional:
    """
    Replace benign region S with attack region S'.

    This performs the graph edit described in Algorithm 1: remove the benign
    aligned region, insert the complete attack region including unmatched
    attack nodes, copy S' internal edges, preserve benign context edges, and
    redirect boundary edges through the alignment map.
    """
    if ig is None:
        return None

    try:
        S_b_set = set(S_b_nodes)
        boundary_context = set()
        for benign_node in S_b_set:
            boundary_context.update(
                g_b.neighborhood(benign_node, order=int(r_hop), mode="all")
            )
        boundary_context -= S_b_set
        inv_pi = {v: w for w, v in pi.items()}
        S_a_set = set(S_a_nodes)
        fidelity_spec = _attack_chain_spec(g_a, S_a_set)
        if fidelity_spec is None:
            return None

        g_mut = ig.Graph(directed=g_b.is_directed())
        for key in g_b.attributes():
            g_mut[key] = g_b[key]
        g_mut["_athena_required_attack_nodes"] = fidelity_spec["required_attack_vertices"]
        g_mut["_athena_required_attack_edges"] = fidelity_spec["required_attack_edges"]
        benign_to_mut: Dict[int, int] = {}
        attack_to_mut: Dict[int, int] = {}

        def add_vertex(
            attrs: dict,
            replaced_region: bool = False,
            in_boundary_context: bool = False,
        ) -> int:
            idx = g_mut.vcount()
            g_mut.add_vertices(1)
            for key, value in attrs.items():
                g_mut.vs[idx][key] = value
            g_mut.vs[idx]["_athena_replaced_region"] = bool(replaced_region)
            g_mut.vs[idx]["_athena_structurally_introduced"] = bool(replaced_region)
            g_mut.vs[idx]["_athena_boundary_context"] = bool(in_boundary_context)
            return idx

        def add_edge(src: int, dst: int, attrs: dict, *, boundary: bool = False) -> None:
            attrs = dict(attrs)
            attrs["_athena_boundary_edge"] = bool(boundary)
            try:
                g_mut.add_edge(src, dst, **attrs)
            except Exception:
                g_mut.add_edge(src, dst)
                new_e = g_mut.es[g_mut.ecount() - 1]
                for key, value in attrs.items():
                    new_e[key] = value

        for v_idx in range(g_b.vcount()):
            if v_idx not in S_b_set:
                benign_to_mut[v_idx] = add_vertex(
                    dict(g_b.vs[v_idx].attributes()),
                    replaced_region=False,
                    in_boundary_context=v_idx in boundary_context,
                )

        for w_idx in S_a_nodes:
            attack_attrs = dict(g_a.vs[w_idx].attributes())
            attack_to_mut[w_idx] = add_vertex(
                attack_attrs,
                replaced_region=True,
            )
            g_mut.vs[attack_to_mut[w_idx]]["_athena_attack_source_vertex"] = int(w_idx)
            g_mut.vs[attack_to_mut[w_idx]]["_athena_attack_original_mutable_attributes"] = {
                key: value for key, value in attack_attrs.items()
                if key in {
                    "properties", "cmdline", "command", "arguments", "path",
                    "srcaddr", "srcport", "dstaddr", "dstport", "address", "port",
                }
            }

        for e_idx in range(g_b.ecount()):
            e = g_b.es[e_idx]
            src_in = e.source in S_b_set
            dst_in = e.target in S_b_set
            if src_in and dst_in:
                continue

            if src_in:
                mapped_src = inv_pi.get(e.source)
                if mapped_src is None:
                    continue
                new_src = attack_to_mut.get(mapped_src)
            else:
                new_src = benign_to_mut.get(e.source)

            if dst_in:
                mapped_dst = inv_pi.get(e.target)
                if mapped_dst is None:
                    continue
                new_dst = attack_to_mut.get(mapped_dst)
            else:
                new_dst = benign_to_mut.get(e.target)

            if new_src is not None and new_dst is not None:
                add_edge(new_src, new_dst, dict(e.attributes()), boundary=src_in != dst_in)

        for e_idx in range(g_a.ecount()):
            e = g_a.es[e_idx]
            if e.source in S_a_set and e.target in S_a_set:
                new_src = attack_to_mut.get(e.source)
                new_dst = attack_to_mut.get(e.target)
                if new_src is not None and new_dst is not None:
                    attrs = dict(e.attributes())
                    attrs["_athena_introduced_edge"] = True
                    attrs["_athena_attack_source_event_id"] = str(
                        attrs.get("event_id", e_idx)
                    )
                    attrs["_athena_attack_source_vertex"] = int(e.source)
                    attrs["_athena_attack_target_vertex"] = int(e.target)
                    add_edge(new_src, new_dst, attrs)

        return g_mut

    except Exception as ex:
        print(f"[StructMut] subgraphreplacefailed: {ex}")
        return None
