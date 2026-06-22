"""
Weisfeiler-Leman subtree kernel implementation

fortwo igraph graph's similarity (paperformula 4) . 
useclass (process/file/socket) , class (read/write/exec/fork/connect) 
andnodeattribute (process, commandrowparameter, filepath, fardegreely) isinitiallabel. 
"""
from __future__ import annotations
from collections import Counter
from typing import List, Optional
import hashlib
import math

try:
    import igraph as ig
except ImportError:
    ig = None


def _node_initial_label(g, v_idx: int) -> str:
    """nodeinitiallabel: entity_type + degreeattribute (from properties take) . 

    name is UUID (graphnot) , mustuse properties takesemanticinfo. 
    properties : process='cmdLine,tgid,path', file='filepath', ='srcaddr,srcport,dstaddr,dstport'. 
    """
    attrs = g. vs [v_idx].attributes()
    entity_type = str(attrs.get("type", "UNK")).lower()
    raw_prop = str(attrs.get("properties", ""))

    #  str(set(...)) : take1meta
    prop = raw_prop.strip()
    if prop.startswith("{") and prop.endswith("}"):
        prop = prop[1:-1].strip()
        if prop.startswith("'") or prop.startswith('"'):
            q = prop[0]
            end = prop.find(q, 1)
            if end > 0:
                prop = prop[1:end]

    # degree: from properties takehasdegree's semanticlabel
    if "process" in entity_type or "subject" in entity_type:
        # process: from properties 's  cmdLine takecommand
        cmd_line = prop.split(",")[0] if prop else ""
        parts = cmd_line.split()
        token = parts[0] if parts else "UNK"
        # takecommand (pathprefixlike /usr/bin/) 
        if "/" in token:
            token = token.rstrip("/").rsplit("/", 1)[-1]
        coarse = token[:16]
    elif "file" in entity_type:
        # file: properties isfilepath, preservedirectorybefore 3 
        fp = prop.strip("{ '\"}")
        segments = fp.split("/")[:4]
        coarse = "/".join(segments) if segments and segments[0] != "" else fp[:24]
    elif "net" in entity_type or "flow" in entity_type or "sock" in entity_type:
        # : properties='srcaddr,srcport,dstaddr,dstport', take
        parts = prop.strip("{ '\"}")  .split(",")
        if len(parts) >= 4:
            coarse = f"port:{parts[3].strip()}"
        elif len(parts) >= 2:
            coarse = f"port:{parts[1].strip()}"
        else:
            coarse = "net"
    else:
        coarse = prop[:12] if prop else "UNK"

    return f"{entity_type}|{coarse}"


def _edge_label(g, e_idx: int) -> str:
    """edgelabel: class"""
    attrs = g.es[e_idx].attributes()
    return str(attrs.get("actions", "UNK"))


def wl_subtree_labels(g, h: int = 3, max_nodes: int = 5000) -> List[Counter]:
    """
    row h  WL iteration, returneach's labelsquaregraphlist. 
    largegraph (>max_nodes) timesamplenode, . 

    Args:
        g: igraph.Graph
        h: WL iterationnumber
        max_nodes: nodenumberexceedthis timesample

    Returns:
        List[Counter]: length h+1, each Counter is's labelfrequency
    """
    import random as _rng
    n = g.vcount()
    if n == 0:
        return [Counter() for _ in range(h + 1)]

    if n > max_nodes:
        sampled = sorted(_rng.sample(range(n), max_nodes))
        g = g.subgraph(sampled)
        n = g.vcount()

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
    """
    twograph's  WL subtree kernel  (normalizeinsideproduct) . 

    K_WL(g1, g2) = sum_i <phi_i(g1), phi_i(g2)> / (||phi(g1)|| * ||phi(g2)||)

    Args:
        g1, g2: igraph.Graph
        h: WL iterationnumber

    Returns:
        float: normalizesimilarity [0, 1]
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
    """from's  WL histogram normalize kernel """
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
    """
    fromattack graphsetinretrieveand benign_graph most similar's  Top-K attack graph (paperformula 4) . 
    supportcachebyattack graph's  WL histogram. 

    Args:
        benign_graph: anchorbenign graph
        attack_graphs: note's attack graphset [(graph, snapshot_idx), ...]
        k: returncount
        h: WL iterationnumber
        _attack_hist_cache: optional's attack graph histogram cache {snapshot_idx: histograms}

    Returns:
        [(graph, snapshot_idx, similarity), ...] similaritydescendingcolumn
    """
    hist_b = wl_subtree_labels(benign_graph, h)

    scored = []
    for ag, sidx in attack_graphs:
        # priority first usecache's attack graph histogram
        if _attack_hist_cache is not None and sidx in _attack_hist_cache:
            hist_a = _attack_hist_cache[sidx]
        else:
            hist_a = wl_subtree_labels(ag, h)
            if _attack_hist_cache is not None:
                _attack_hist_cache[sidx] = hist_a
        sim = _kernel_from_histograms(hist_b, hist_a)
        scored.append((ag, sidx, sim))

    scored.sort(key=lambda x: -x[2])
    return scored[:k]
