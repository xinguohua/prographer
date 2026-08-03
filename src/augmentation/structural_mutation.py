"""Paper §IV.B - Structural mutation (Algorithm 1).

For each benign anchor G_b:
1. WL-subtree retrieval yields the Top-K most similar attack snapshots.
2. For each attack snapshot, locate attack nodes, choose the best seed
   pairing by lightweight matching, then BFS-align an attack region S'.
3. Replace the aligned region of G_b with S' to produce the mutated graph.
"""
from __future__ import annotations
from collections import deque
from typing import List, Tuple, Dict, Optional
import random

try:
    import igraph as ig
except ImportError:
    ig = None


def _token_set(g, v_idx: int) -> set:
    """takenode's  token set, for Jaccard similarity"""
    attrs = g. vs [v_idx].attributes()
    tokens = set()
    for k, v in attrs.items():
        if k in ("label",):
            continue
        tokens.add(f"{k}:{v}")
    return tokens


def _jaccard_tok(tokens_a: set, tokens_b: set) -> float:
    if not tokens_a and not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return inter / union if union > 0 else 0.0


def aligned_region_search(
    g_b, g_a,
    max_region_size: int = 32,
) -> List[Tuple[list, list, dict, float]]:
    """
    BFS alignsubgraphregion. 

    optimize: first matchmost startto, againonlytomost to BFS. 
    from O(seeds × procs_b) time BFS to O(seeds) time BFS. 
    """
    attack_nodes_a = set()
    for i in range(g_a.vcount()):
        if g_a. vs [i].attributes().get("label", 0) == 1:
            attack_nodes_a.add(i)

    if attack_nodes_a:
        seed_set = set()
        for a_idx in attack_nodes_a:
            seed_set.add(a_idx)
            for nb in g_a.neighbors(a_idx, mode="all"):
                t = str(g_a. vs [nb].attributes().get("type", "")).lower()
                if "process" in t or "subject" in t:
                    seed_set.add(nb)
        attack_seeds_a = list(seed_set)
    else:
        attack_seeds_a = [
            i for i in range(g_a.vcount())
            if "process" in str(g_a. vs [i].attributes().get("type", "")).lower()
               or "subject" in str(g_a. vs [i].attributes().get("type", "")).lower()
        ]

    type_to_nodes_b: Dict[str, List[int]] = {}
    for i in range(g_b.vcount()):
        t = str(g_b. vs [i].attributes().get("type", "")).lower()
        type_to_nodes_b.setdefault(t, []).append(i)

    if not attack_seeds_a:
        return []

    if len(attack_seeds_a) > 10:
        attack_first = [s for s in attack_seeds_a if s in attack_nodes_a]
        others = [s for s in attack_seeds_a if s not in attack_nodes_a]
        attack_seeds_a = (attack_first + others)[:10]

    candidates = []

    for w in attack_seeds_a:
        # ---- match:  G_b inmost tonode (O(n) Jaccard  vs more, not BFS)  ----
        w_type = str(g_a. vs [w].attributes().get("type", "")).lower()
        w_tokens = _token_set(g_a, w)

        # classnodein Jaccard highest's 
        same_type = type_to_nodes_b.get(w_type, [])
        if not same_type:
            for t_key in type_to_nodes_b:
                if "process" in t_key or "subject" in t_key:
                    same_type = type_to_nodes_b[t_key]
                    break
        if not same_type:
            continue

        if len(same_type) > 20:
            candidates_b = random.sample(same_type, 20)
        else:
            candidates_b = same_type

        best_v = max(candidates_b, key=lambda u: _jaccard_tok(_token_set(g_b, u), w_tokens))

        # ---- onlytomost to1time BFS ----
        v = best_v
        pi = {w: v}
        S_b = {v}
        S_a = {w}
        queue = deque([w])

        while queue and len(S_a) < max_region_size:
            w_c = queue.popleft()
            v_c = pi[w_c]

            neighbors_a = set(g_a.neighbors(w_c, mode="all")) - S_a
            neighbors_b = set(g_b.neighbors(v_c, mode="all")) - S_b

            neighbors_a_sorted = sorted(
                neighbors_a,
                key=lambda n: (1 if n in attack_nodes_a else 0),
                reverse=True,
            )

            for w_prime in neighbors_a_sorted:
                type_a = str(g_a. vs [w_prime].attributes().get("type", ""))

                type_consistent = [
                    u for u in neighbors_b
                    if str(g_b. vs [u].attributes().get("type", "")) == type_a
                ]

                if type_consistent:
                    tokens_a = _token_set(g_a, w_prime)
                    best_u = max(type_consistent,
                                 key=lambda u: _jaccard_tok(_token_set(g_b, u), tokens_a))
                    v_prime = best_u
                elif neighbors_b:
                    tokens_a = _token_set(g_a, w_prime)
                    v_prime = max(neighbors_b,
                                  key=lambda u: _jaccard_tok(_token_set(g_b, u), tokens_a))
                else:
                    continue

                pi[w_prime] = v_prime
                S_a.add(w_prime)
                S_b.add(v_prime)
                neighbors_b.discard(v_prime)
                queue.append(w_prime)

        n_matched = sum(
            1 for w_j, v_j in pi.items()
            if str(g_a. vs [w_j].attributes().get("type", "")) == str(g_b. vs [v_j].attributes().get("type", ""))
        )
        rho = n_matched / len(pi) if pi else 0.0

        n_attack_covered = sum(1 for w_j in pi.keys() if w_j in attack_nodes_a)
        attack_ratio = n_attack_covered / max(1, len(attack_nodes_a)) if attack_nodes_a else 0.0

        score = attack_ratio * 2.0 + rho

        if len(pi) >= 2:
            candidates.append((list(S_b), list(S_a), dict(pi), score))

    candidates.sort(key=lambda x: -x[3])
    seen_regions = []
    deduped = []
    for cand in candidates:
        s_b_set = set(cand[0])
        overlap = any(len(s_b_set & seen) > len(s_b_set) * 0.5 for seen in seen_regions)
        if not overlap:
            deduped.append(cand)
            seen_regions.append(s_b_set)
    return deduped


def subgraph_replacement(
    g_b, g_a,
    S_b_nodes: list,
    S_a_nodes: list,
    pi: dict,
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
        inv_pi = {v: w for w, v in pi.items()}
        S_a_set = set(S_a_nodes)

        g_mut = ig.Graph(directed=g_b.is_directed())
        benign_to_mut: Dict[int, int] = {}
        attack_to_mut: Dict[int, int] = {}

        def add_vertex(attrs: dict, replaced_region: bool = False) -> int:
            idx = g_mut.vcount()
            g_mut.add_vertices(1)
            for key, value in attrs.items():
                g_mut.vs[idx][key] = value
            g_mut.vs[idx]["_athena_replaced_region"] = bool(replaced_region)
            return idx

        def add_edge(src: int, dst: int, attrs: dict) -> None:
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
                )

        for w_idx in S_a_nodes:
            attack_to_mut[w_idx] = add_vertex(
                dict(g_a.vs[w_idx].attributes()),
                replaced_region=True,
            )

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
                add_edge(new_src, new_dst, dict(e.attributes()))

        for e_idx in range(g_a.ecount()):
            e = g_a.es[e_idx]
            if e.source in S_a_set and e.target in S_a_set:
                new_src = attack_to_mut.get(e.source)
                new_dst = attack_to_mut.get(e.target)
                if new_src is not None and new_dst is not None:
                    add_edge(new_src, new_dst, dict(e.attributes()))

        return g_mut

    except Exception as ex:
        print(f"[StructMut] subgraphreplacefailed: {ex}")
        return None
