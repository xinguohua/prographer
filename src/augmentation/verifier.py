"""Paper §IV.C - Unified verification (four checks).

Each mutated graph G~ must pass four checks before it is admitted as a hard
negative for contrastive learning:

1. Operation legality - every edge whose endpoint sits in the replaced
   region must use an action that was observed for that source-entity type in
   the historical benign graphs.
2. Attribute feasibility - every replaced node's attributes must lie in
   the observed attribute vocabulary for its node type.
3. Imperceptibility - no boundary-touching edge may carry a
   user-perceivable action (GUI create / show / notify / alert / prompt /
   dialog), which would otherwise tip off a human observer.
4. Hardness - the WL similarity between the mutated graph and its
   benign anchor must lie within [delta_h, delta_h_upper]: similar enough to
   be a hard negative, distinct enough to carry attack signal.

A mutation is *admitted* iff all four checks pass.
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    import igraph as ig  # noqa: F401
except ImportError:
    ig = None


PERCEIVABLE_BLACKLIST = {
    ("process", "gui_create"),
    ("process", "gui_show"),
    ("process", "notify"),
    ("process", "alert"),
    ("process", "prompt"),
    ("process", "dialog"),
}


def _norm(value) -> str:
    return str(value).strip().lower()


def _entity_key(attrs: dict) -> str:
    for key in ("uuid", "name", "properties", "cmdline", "path"):
        value = attrs.get(key)
        if value not in (None, ""):
            return f"{key}:{_norm(value)}"
    return f"type:{_norm(attrs.get('type', ''))}"


def build_historical_profiles(benign_graphs: list) -> Tuple[
    Dict[str, Set[str]],
    Dict[str, Dict[str, Set[str]]],
]:
    """Scan the benign-training graphs to produce two lookup tables:

    - ``entity_ops``  : source entity signature -> set of actions observed,
    - ``type_attrs``  : node type -> attribute name -> observed values.

    These tables feed the operation-legality and attribute-feasibility
    checks below.
    """
    entity_ops: Dict[str, Set[str]] = defaultdict(set)
    type_attrs: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for g, _ in benign_graphs:
        for e_idx in range(g.ecount()):
            e = g.es[e_idx]
            src_attrs = g.vs[e.source].attributes()
            action = str(e.attributes().get("actions", ""))
            entity_ops[_entity_key(src_attrs)].add(action)

        for v_idx in range(g.vcount()):
            attrs = g.vs[v_idx].attributes()
            vtype = _norm(attrs.get("type", ""))
            for attr_name, attr_val in attrs.items():
                if attr_name in {"label"} or attr_val in (None, ""):
                    continue
                type_attrs[vtype][attr_name].add(_norm(attr_val))

    return dict(entity_ops), {k: dict(v) for k, v in type_attrs.items()}


def check_operation_legality(
    g_mut,
    replaced_nodes: Set[int],
    entity_ops: Dict[str, Set[str]],
) -> bool:
    """Check 1 (paper §IV.C). Returns ``True`` when every
    boundary-touching edge uses an action that was previously observed for
    its concrete source entity in ``entity_ops``."""

    if not entity_ops:
        return True
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        if e.source not in replaced_nodes and e.target not in replaced_nodes:
            continue
        action = str(e.attributes().get("actions", ""))
        src_key = _entity_key(g_mut.vs[e.source].attributes())
        observed = entity_ops.get(src_key)
        if observed is None or action not in observed:
            return False
    return True


def check_attribute_feasibility(
    g_mut,
    replaced_nodes: Set[int],
    type_attrs: Dict[str, Dict[str, Set[str]]],
) -> bool:
    """Check 2 (paper §IV.C). Returns ``True`` when every
    replaced node's checked attributes are compatible with that node type in
    ``type_attrs``. Each attribute is checked under its own name; a match in an
    unrelated attribute no longer admits the mutation."""

    if not type_attrs:
        return True
    for v_idx in replaced_nodes:
        attrs = g_mut.vs[v_idx].attributes()
        vtype = _norm(attrs.get("type", ""))
        observed_by_attr = type_attrs.get(vtype)
        if observed_by_attr is None:
            return False
        checked = 0
        for attr_name, attr_val in attrs.items():
            if attr_name in {"label"} or attr_val in (None, ""):
                continue
            observed_values = observed_by_attr.get(attr_name)
            checked += 1
            if observed_values is None or _norm(attr_val) not in observed_values:
                return False
        if checked == 0:
            return False
    return True


def check_imperceptibility(g_mut, replaced_nodes: Set[int]) -> bool:
    """Check 3 (paper §IV.C). Rejects any mutation whose boundary
    introduces a user-visible action that a human observer would notice."""
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        if e.source not in replaced_nodes and e.target not in replaced_nodes:
            continue
        action = str(e.attributes().get("actions", "")).lower()
        src_type = str(g_mut. vs [e.source].attributes().get("type", "")).lower()
        for etype, op in PERCEIVABLE_BLACKLIST:
            if etype in src_type and op in action:
                return False
    return True


def check_hardness(g_mut, g_anchor, delta_h: float = 0.3, delta_h_upper: float = 0.95) -> bool:
    """Check 4 (paper §IV.C). Rejects mutations whose WL
    similarity to the anchor falls outside ``[delta_h, delta_h_upper]``."""
    from .subgraph_retrieval import wl_kernel
    sim = wl_kernel(g_mut, g_anchor, h=3)
    return delta_h <= sim <= delta_h_upper


def verify_mutation(
    g_mut,
    g_anchor,
    replaced_nodes: Set[int],
    entity_ops: Dict[str, Set[str]],
    type_attrs: Dict[str, Dict[str, Set[str]]],
    delta_h: float = 0.3,
    delta_h_upper: float = 0.95,
) -> Tuple[bool, List[str]]:
    """Run all four checks. Returns ``(passed, failed_checks)``."""
    failed: List[str] = []

    if not check_operation_legality(g_mut, replaced_nodes, entity_ops):
        failed.append("operation_legality")
    if not check_attribute_feasibility(g_mut, replaced_nodes, type_attrs):
        failed.append("attribute_feasibility")
    if not check_imperceptibility(g_mut, replaced_nodes):
        failed.append("imperceptibility")
    if not check_hardness(g_mut, g_anchor, delta_h=delta_h, delta_h_upper=delta_h_upper):
        failed.append("hardness")

    return len(failed) == 0, failed
