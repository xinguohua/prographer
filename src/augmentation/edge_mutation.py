"""Paper §IV.B / Supp B - LLM-guided boundary edge mutation.

After structural replacement substitutes the aligned attack region ``S'`` into
the benign anchor ``G+``, the only edges that may be edited are the boundary
edges between ``S'`` and the surrounding context ``C``. Internal edges of
``S'`` (the attack causal chain) and internal edges of ``C`` are not editable.
For every candidate boundary edge the LLM is asked to decide one of:

- ``ADD``    - add the proposed new edge if the cross-boundary connection is
  plausible in the benign workflow of ``C``;
- ``REMOVE`` - remove the existing edge if it would otherwise expose the attack
  or violate type/operation feasibility;
- ``KEEP``   - leave the edge unchanged.

The prompt template lives at ``prompts/edge_mutation.txt``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple

try:
    import igraph as ig  # noqa: F401
except ImportError:
    ig = None


PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "edge_mutation.txt"


def _load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _format_triple(g, edge) -> Tuple[str, str, str]:
    """Format an edge as ``(src_type:attr, op, dst_type:attr)`` for the prompt."""
    src_attrs = g. vs [edge.source].attributes()
    dst_attrs = g. vs [edge.target].attributes()
    src_type = str(src_attrs.get("type", "")).lower()
    dst_type = str(dst_attrs.get("type", "")).lower()
    src_prop = str(src_attrs.get("properties", "") or src_attrs.get("label", ""))
    dst_prop = str(dst_attrs.get("properties", "") or dst_attrs.get("label", ""))
    op = str(edge.attributes().get("actions", ""))
    return (f"{src_type}:{src_prop}", op, f"{dst_type}:{dst_prop}")


def collect_boundary_edges(g, replaced_nodes: Set[int]) -> List[int]:
    """Return edge indices whose endpoints span the boundary between ``S'``
    (the replaced region) and the surrounding context ``C``."""
    boundary: List[int] = []
    for e_idx in range(g.ecount()):
        e = g.es[e_idx]
        src_in = e.source in replaced_nodes
        dst_in = e.target in replaced_nodes
        if src_in != dst_in:
            boundary.append(e_idx)
    return boundary


def propose_candidate_new_edges(
    g,
    replaced_nodes: Set[int],
    max_candidates: int = 16,
) -> List[Tuple[int, int, str]]:
    """Suggest plausible new boundary edges to consider adding.

    Each candidate is ``(source_idx, target_idx, action)`` where exactly one
    endpoint is in ``replaced_nodes``. Action defaults to a generic ``connect``
    verb so the LLM judges feasibility on the typed triple alone.
    """
    if not replaced_nodes:
        return []
    context_nodes = [v for v in range(g.vcount()) if v not in replaced_nodes]
    if not context_nodes:
        return []
    existing: Set[Tuple[int, int]] = set()
    for e in g.es:
        existing.add((e.source, e.target))

    candidates: List[Tuple[int, int, str]] = []
    for v_s in sorted(replaced_nodes):
        for v_c in context_nodes:
            if (v_s, v_c) in existing or (v_c, v_s) in existing:
                continue
            candidates.append((v_s, v_c, "connect"))
            if len(candidates) >= max_candidates:
                return candidates
    return candidates


def _build_prompt(
    template: str,
    remove_triples: List[Tuple[str, str, str]],
    add_triples: List[Tuple[str, str, str]],
) -> str:
    def _fmt(triples: List[Tuple[str, str, str]]) -> str:
        if not triples:
            return "[]"
        return "[" + ", ".join(
            f"({s}, {op}, {d})" for s, op, d in triples
        ) + "]"

    return (
        template
        .replace("{{REMOVE_TRIPLES}}", _fmt(remove_triples))
        .replace("{{ADD_TRIPLES}}", _fmt(add_triples))
    )


def _parse_llm_response(text: str, num_remove: int, num_add: int) -> List[str]:
    """Parse the LLM JSON response into a list of actions, one per edge in
    ``remove`` order followed by ``add`` order. Unrecognised actions fall
    back to ``KEEP`` (remove) / ``DROP`` (add)."""
    actions: List[str] = []
    try:
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            raise ValueError("no JSON array in response")
        items = json.loads(match.group(0))
    except (ValueError, json.JSONDecodeError):
        items = []

    by_id = {int(it.get("edge_id", i)): str(it.get("action", "")).upper()
             for i, it in enumerate(items) if isinstance(it, dict)}

    total = num_remove + num_add
    for i in range(total):
        a = by_id.get(i, "")
        if i < num_remove:
            actions.append(a if a in {"REMOVE", "KEEP"} else "KEEP")
        else:
            actions.append(a if a in {"ADD", "DROP"} else "DROP")
    return actions


def apply_edge_mutation_llm(
    g,
    replaced_nodes: Set[int],
    llm_fn: Optional[Callable[[str], str]] = None,
    max_add_candidates: int = 16,
):
    """Apply LLM-guided ADD / REMOVE / KEEP decisions on boundary edges.

    Returns ``(g_out, decisions)`` where ``g_out`` is the mutated graph and
    ``decisions`` is the parallel action list for logging. If ``llm_fn`` is
    ``None``, the graph is returned unchanged.
    """
    if llm_fn is None or not replaced_nodes:
        return g, []

    remove_ids = collect_boundary_edges(g, replaced_nodes)
    add_candidates = propose_candidate_new_edges(g, replaced_nodes, max_add_candidates)

    if not remove_ids and not add_candidates:
        return g, []

    remove_triples = [_format_triple(g, g.es[ei]) for ei in remove_ids]
    add_triples = [(
        f"{g. vs [s].attributes().get('type', '')}:{g. vs [s].attributes().get('properties', '')}",
        op,
        f"{g. vs [t].attributes().get('type', '')}:{g. vs [t].attributes().get('properties', '')}",
    ) for (s, t, op) in add_candidates]

    template = _load_prompt_template()
    prompt = _build_prompt(template, remove_triples, add_triples)
    raw = llm_fn(prompt)
    actions = _parse_llm_response(raw, len(remove_ids), len(add_candidates))

    to_remove: List[int] = []
    for i, ei in enumerate(remove_ids):
        if actions[i] == "REMOVE":
            to_remove.append(ei)

    to_add: List[Tuple[int, int, str]] = []
    for j, cand in enumerate(add_candidates):
        if actions[len(remove_ids) + j] == "ADD":
            to_add.append(cand)

    g_out = g.copy()
    if to_remove:
        g_out.delete_edges(sorted(to_remove, reverse=True))
    for (s, t, op) in to_add:
        g_out.add_edge(s, t, actions=op)

    return g_out, actions
