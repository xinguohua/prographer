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
import inspect
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Set, Tuple

from .property_adapter import semantic_summary

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
    src_prop = semantic_summary(src_attrs.get("properties", ""), src_type)
    dst_prop = semantic_summary(dst_attrs.get("properties", ""), dst_type)
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
    max_candidates: Optional[int] = None,
) -> List[Tuple[int, int, str]]:
    """Suggest plausible new boundary edges to consider adding.

    Each candidate is ``(source_idx, target_idx, action)`` where exactly one
    endpoint is in ``replaced_nodes``. Candidate actions are restricted to
    operation types already observed on the replacement/boundary context.
    """
    if not replaced_nodes:
        return []
    if "_athena_boundary_context" not in g.vs.attributes():
        return []
    context_nodes = [
        vertex
        for vertex, flag in enumerate(g.vs["_athena_boundary_context"])
        if bool(flag) and vertex not in replaced_nodes
    ]
    if not context_nodes:
        return []
    existing: Set[Tuple[int, int]] = set()
    for e in g.es:
        existing.add((e.source, e.target))

    operation_types = sorted({
        str(edge.attributes().get("actions", ""))
        for edge in g.es
        if str(edge.attributes().get("actions", ""))
        and (
            edge.source in replaced_nodes
            or edge.target in replaced_nodes
            or edge.source in context_nodes
            or edge.target in context_nodes
        )
    })
    if not operation_types:
        return []
    candidates: List[Tuple[int, int, str]] = []
    for v_s in sorted(replaced_nodes):
        for v_c in context_nodes:
            for src, dst in ((v_s, v_c), (v_c, v_s)):
                if (src, dst) in existing:
                    continue
                for operation in operation_types:
                    candidates.append((src, dst, operation))
                    if max_candidates is not None and len(candidates) >= int(max_candidates):
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
    """Strictly validate one LLM decision for every candidate edge."""
    try:
        match = re.search(r"\[.*\]", text, re.S)
        if not match:
            raise ValueError("no JSON array in response")
        items = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("edge mutation response is not valid JSON") from exc
    total = num_remove + num_add
    if not isinstance(items, list) or len(items) != total:
        raise ValueError(f"expected {total} edge decisions, received {len(items) if isinstance(items, list) else 'non-list'}")
    by_id = {}
    for item in items:
        if not isinstance(item, dict) or "edge_id" not in item or "action" not in item:
            raise ValueError("each edge decision requires edge_id and action")
        edge_id = int(item["edge_id"])
        if edge_id in by_id or edge_id < 0 or edge_id >= total:
            raise ValueError(f"invalid or duplicate edge_id: {edge_id}")
        action = str(item["action"]).upper()
        allowed = {"REMOVE", "KEEP"} if edge_id < num_remove else {"ADD", "KEEP"}
        if action not in allowed:
            raise ValueError(f"invalid action {action!r} for edge {edge_id}")
        by_id[edge_id] = action
    if set(by_id) != set(range(total)):
        raise ValueError("edge decisions do not cover every candidate exactly once")

    actions: List[str] = []
    for i in range(total):
        actions.append(by_id[i])
    return actions


def apply_edge_mutation_llm(
    g,
    replaced_nodes: Set[int],
    llm_fn: Optional[Callable[[str], str]] = None,
    max_add_candidates: Optional[int] = None,
    max_candidates_per_call: int = 64,
    max_prompt_chars: int = 6000,
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
        f"{g. vs [s].attributes().get('type', '')}:"
        f"{semantic_summary(g. vs [s].attributes().get('properties', ''), g. vs [s].attributes().get('type', ''))}",
        op,
        f"{g. vs [t].attributes().get('type', '')}:"
        f"{semantic_summary(g. vs [t].attributes().get('properties', ''), g. vs [t].attributes().get('type', ''))}",
    ) for (s, t, op) in add_candidates]

    template = _load_prompt_template()
    total_candidates = len(remove_ids) + len(add_candidates)
    batch_size = max(1, int(max_candidates_per_call))
    prompt_budget = max(512, int(max_prompt_chars))
    actions = [""] * total_candidates
    items = [
        ("remove", index, triple)
        for index, triple in enumerate(remove_triples)
    ] + [
        ("add", len(remove_ids) + index, triple)
        for index, triple in enumerate(add_triples)
    ]
    batches = []
    pending = []
    for item in items:
        proposed = pending + [item]
        proposed_remove = [row[2] for row in proposed if row[0] == "remove"]
        proposed_add = [row[2] for row in proposed if row[0] == "add"]
        proposed_prompt = _build_prompt(template, proposed_remove, proposed_add)
        if pending and (len(proposed) > batch_size or len(proposed_prompt) > prompt_budget):
            batches.append(pending)
            pending = [item]
        else:
            pending = proposed
        single_prompt = _build_prompt(
            template,
            [pending[0][2]] if pending[0][0] == "remove" else [],
            [pending[0][2]] if pending[0][0] == "add" else [],
        )
        if len(single_prompt) > prompt_budget:
            raise ValueError("edge candidate cannot fit the configured prompt character budget")
    if pending:
        batches.append(pending)
    batch_count = len(batches)
    try:
        signature = inspect.signature(llm_fn)
        accepts_metadata = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_metadata = False
    for batch_index, batch in enumerate(batches, 1):
        batch_remove = [item for item in batch if item[0] == "remove"]
        batch_add = [item for item in batch if item[0] == "add"]
        prompt = _build_prompt(
            template,
            [item[2] for item in batch_remove],
            [item[2] for item in batch_add],
        )
        metadata = {
            "batch_index": batch_index,
            "batch_count": batch_count,
            "candidate_count": len(batch),
            "total_candidate_count": total_candidates,
            "prompt_chars": len(prompt),
            "prompt_char_budget": prompt_budget,
        }
        try:
            raw = llm_fn(prompt, **metadata) if accepts_metadata else llm_fn(prompt)
        except Exception as exc:
            print(f"[EdgeMut] LLM call failed after retries: {exc}")
            return None, []
        try:
            batch_actions = _parse_llm_response(raw, len(batch_remove), len(batch_add))
        except (TypeError, ValueError) as exc:
            print(f"[EdgeMut] rejected LLM response: {exc}")
            return None, []
        ordered_batch = batch_remove + batch_add
        for item, action in zip(ordered_batch, batch_actions):
            actions[item[1]] = action
    if any(not action for action in actions):
        raise RuntimeError("edge mutation batching lost a candidate decision")

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
        g_out.add_edge(
            s,
            t,
            actions=op,
            _athena_edge_mutated=True,
            _athena_boundary_edge=True,
        )

    return g_out, actions
