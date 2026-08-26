"""Paper §IV.C - Unified verification (four checks).

Each mutated graph G~ must pass four checks before it is admitted as a hard
negative for contrastive learning:

1. Operation legality - every edge whose endpoint sits in the replaced
   region must use an action that was observed for that source entity in
   the historical benign graphs.
2. Attribute feasibility - every replaced node's attributes must lie in
   the observed attribute vocabulary for its node type.
3. Imperceptibility - no boundary-touching edge may carry a
   user-perceivable action (GUI create / show / notify / alert / prompt /
   dialog), which would otherwise tip off a human observer.
4. Hardness - the WL similarity between the mutated graph and its benign
   anchor must be at least ``delta_h``.

A mutation is *admitted* iff all four checks pass.
"""
from __future__ import annotations
from collections import defaultdict
import re
from typing import Dict, List, Set, Tuple

from .property_adapter import process_records, process_view, typed_fields

try:
    import igraph as ig  # noqa: F401
except ImportError:
    ig = None


_COMMON_PERCEIVABLE_ACTIONS = {
    "gui": {"gui_create", "gui_show", "window_create", "window_show"},
    "notification": {"notify", "notification", "show_notification", "toast", "alert"},
    "auth_prompt": {"auth_prompt", "credential_prompt", "password_prompt", "show_dialog"},
    "audible": {"audible", "beep", "play_sound", "audio_play"},
    "visual": {"visual", "flash", "screen_write", "display", "render"},
}

# Explicit raw-action families for the four released provenance sources.  The
# common canonical spellings are augmented with each source's event prefix.
PERCEIVABLE_ACTIONS = {
    "darpa_e3": {
        category: values | {f"event_{value}" for value in values}
        for category, values in _COMMON_PERCEIVABLE_ACTIONS.items()
    },
    "darpa_e5": {
        category: values | {f"event_{value}" for value in values}
        for category, values in _COMMON_PERCEIVABLE_ACTIONS.items()
    },
    "optc": {category: set(values) for category, values in _COMMON_PERCEIVABLE_ACTIONS.items()},
    "atlas": {category: set(values) for category, values in _COMMON_PERCEIVABLE_ACTIONS.items()},
}

PERCEIVABLE_TARGET_MARKERS = {
    "gui": {"gui", "window", "dialog"},
    "notification": {"notification", "notify", "toast", "alert"},
    "auth_prompt": {"credential", "password", "auth", "prompt", "pam"},
    "audible": {"audio", "audible", "speaker", "sound", "beep", "dev/audio"},
    "visual": {"display", "screen", "visual", "framebuffer", "dev/tty"},
}

MUTABLE_ATTRIBUTE_NAMES = {
    "properties", "cmdline", "command", "arguments", "path",
    "srcaddr", "srcport", "dstaddr", "dstport", "address", "port",
}


def _norm(value) -> str:
    return str(value).strip().lower()


def _entity_keys(attrs: dict) -> List[str]:
    """Type + relevant semantics, deliberately excluding UUID/name identity."""
    vtype = _norm(attrs.get("type", ""))
    if "process" in vtype or "subject" in vtype:
        keys = []
        raw = attrs.get("properties") or attrs.get("cmdline") or attrs.get("command") or ""
        for view in process_records(raw):
            keys.extend(_process_record_keys(vtype, view))
        return keys or [f"type:{vtype}|class:process-unknown"]
    if "file" in vtype:
        value = attrs.get("path") or attrs.get("properties") or ""
        clean = _norm(_clean_serialized_value(value)).rsplit("/", 1)[-1]
        suffix = clean.rsplit(".", 1)[-1] if "." in clean else "no-extension"
        return [f"type:{vtype}|class:file:{suffix}"]
    if any(token in vtype for token in ("net", "flow", "sock")):
        value = attrs.get("address") or attrs.get("dstaddr") or attrs.get("properties") or ""
        clean = _norm(_clean_serialized_value(value))
        scope = "private" if clean.startswith(("10.", "127.", "192.168.", "172.16.")) else "external"
        return [f"type:{vtype}|class:network:{scope}"]
    value = attrs.get("properties") or ""
    return [f"type:{vtype}|semantic:{_norm(_clean_serialized_value(value))}"]


def _process_record_keys(vtype: str, view: dict) -> List[str]:
    keys = []
    if view["command"] and view["arguments"]:
        keys.append(
            f"type:{vtype}|command:{_norm(view['command'])}"
            f"|arguments:{_norm(view['arguments'])}"
        )
    if view["command"]:
        keys.append(f"type:{vtype}|command:{_norm(view['command'])}")
        path_class = _norm(view["path"]).rsplit("/", 1)[-1]
        keys.append(f"type:{vtype}|command:{_norm(view['command'])}|path-class:{path_class}")
    if view["arguments"]:
        keys.append(f"type:{vtype}|arguments:{_norm(view['arguments'])}")
    return keys


def _entity_key(attrs: dict) -> str:
    return _entity_keys(attrs)[0]


def _is_internal_attr(attr_name: str) -> bool:
    return str(attr_name).startswith("_athena_")


def _attribute_components(value) -> Set[str]:
    """Split one atomic field into independently verifiable components."""
    return set(re.findall(r"[a-z0-9_./:@-]+", _norm(value)))


def _clean_serialized_value(value) -> str:
    text = str(value).strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1].strip()
    return text.strip("'\"")


def _typed_attribute_fields(vtype: str, attr_name: str, value) -> Dict[str, Set[str]]:
    """Split structured ``properties`` without crossing entity-field boundaries."""
    prefix = str(attr_name)
    text = _clean_serialized_value(value)
    if prefix == "properties":
        return typed_fields(value, vtype)
    return {prefix: {_norm(text)} if text else set()}


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
            if int(src_attrs.get("label", 0) or 0) == 1:
                continue
            vtype = _norm(src_attrs.get("type", ""))
            records = process_records(src_attrs.get("properties", ""))
            structured = any(row.get("action") or row.get("role") for row in records)
            if structured:
                for row in records:
                    if row.get("role") not in {"", "actor"} or not row.get("action"):
                        continue
                    for key in _process_record_keys(vtype, row):
                        entity_ops[key].add(str(row["action"]))
                continue
            action = str(e.attributes().get("actions", ""))
            for key in _entity_keys(src_attrs):
                entity_ops[key].add(action)

        for v_idx in range(g.vcount()):
            attrs = g.vs[v_idx].attributes()
            if int(attrs.get("label", 0) or 0) == 1:
                continue
            vtype = _norm(attrs.get("type", ""))
            for attr_name, attr_val in attrs.items():
                if attr_name not in MUTABLE_ATTRIBUTE_NAMES or attr_val in (None, ""):
                    continue
                for field_name, components in _typed_attribute_fields(vtype, attr_name, attr_val).items():
                    type_attrs[vtype][field_name].update(components)

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
        return False
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        edge_attrs = e.attributes()
        if not (
            bool(edge_attrs.get("_athena_boundary_edge", False))
            or bool(edge_attrs.get("_athena_edge_mutated", False))
            or bool(edge_attrs.get("_athena_introduced_edge", False))
        ):
            continue
        action = str(e.attributes().get("actions", ""))
        src_attrs = g_mut.vs[e.source].attributes()
        keys = _entity_keys(src_attrs)
        source_type = _norm(src_attrs.get("type", ""))
        records = process_records(src_attrs.get("properties", ""))
        edge_event_id = str(edge_attrs.get("event_id", "") or "")
        event_bound = [
            row for row in records
            if edge_event_id and row.get("event_id") == edge_event_id
            and row.get("role") in {"", "actor"}
        ]
        if event_bound:
            keys = [
                key for row in event_bound for key in _process_record_keys(source_type, row)
            ]
        elif edge_event_id and any(row.get("event_id") for row in records) and not (
            bool(edge_attrs.get("_athena_boundary_edge", False))
            or bool(edge_attrs.get("_athena_edge_mutated", False))
        ):
            return False
        changed_raw = src_attrs.get("_athena_semantic_changed_fields", [])
        changed = set(
            changed_raw if isinstance(changed_raw, (list, tuple, set)) else [changed_raw]
        )
        strategy = str(src_attrs.get("_athena_semantic_strategy", ""))
        if strategy == "replacement" and "command_name" in changed:
            keys = [key for key in keys if "|command:" in key and "|arguments:" not in key]
        elif strategy == "replacement" and changed == {"arguments"}:
            keys = [key for key in keys if "|arguments:" in key and "|command:" not in key]
        observed = set()
        for key in keys:
            if key in entity_ops:
                observed = set(entity_ops[key])
                break
        if action not in observed:
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
        return False
    for v_idx in replaced_nodes:
        attrs = g_mut.vs[v_idx].attributes()
        raw_changed = attrs.get("_athena_semantic_changed_fields", [])
        changed_fields = set(raw_changed if isinstance(raw_changed, (list, tuple, set)) else [raw_changed])
        if not changed_fields and not bool(attrs.get("_athena_semantic_modified", False)):
            continue
        vtype = _norm(attrs.get("type", ""))
        observed_by_attr = type_attrs.get(vtype)
        if observed_by_attr is None:
            return False
        original_attrs = attrs.get("_athena_attack_original_mutable_attributes", {})
        if not isinstance(original_attrs, dict):
            original_attrs = {}
        strategy = str(attrs.get("_athena_semantic_strategy", ""))
        if strategy == "replacement" and "command_name" in changed_fields:
            old_args = process_view(original_attrs.get("properties", ""))["arguments"]
            current_args = process_view(attrs.get("properties", ""))["arguments"]
            exact_positions = [
                match.start() for match in re.finditer(re.escape(old_args), current_args)
                if (match.start() == 0 or current_args[match.start() - 1].isspace())
                and (match.end() == len(current_args) or current_args[match.end()].isspace())
            ] if old_args else [0]
            if not exact_positions:
                return False
        original_fields: Dict[str, Set[str]] = defaultdict(set)
        for attr_name, attr_val in original_attrs.items():
            for field_name, components in _typed_attribute_fields(vtype, attr_name, attr_val).items():
                original_fields[field_name].update(components)
        checked = 0
        for attr_name, attr_val in attrs.items():
            if attr_name not in MUTABLE_ATTRIBUTE_NAMES or attr_val in (None, ""):
                continue
            for field_name, components in _typed_attribute_fields(vtype, attr_name, attr_val).items():
                semantic_field = {
                    "properties.command": "command_name",
                    "properties.arguments": "arguments",
                }.get(field_name, attr_name)
                if changed_fields and semantic_field not in changed_fields and attr_name not in changed_fields:
                    continue
                if field_name == "properties.arguments" and strategy == "extension":
                    continue
                if (
                    field_name == "properties.arguments"
                    and strategy == "replacement"
                    and "command_name" in changed_fields
                ):
                    original_args = original_fields.get(field_name, set())
                    current_args = next(iter(components), "")
                    preserved = any(
                        current_args[index:index + len(value)] == value
                        and (index == 0 or current_args[index - 1].isspace())
                        and (index + len(value) == len(current_args) or current_args[index + len(value)].isspace())
                        for value in original_args
                        for index in range(len(current_args) - len(value) + 1)
                    )
                    if original_args and not preserved:
                        return False
                    if preserved:
                        continue
                new_components = components - original_fields.get(field_name, set())
                if not new_components:
                    continue
                checked += 1
                observed_values = observed_by_attr.get(field_name)
                if observed_values is None or not new_components.issubset(observed_values):
                    return False
    return True


def _dataset_family(g_mut) -> str:
    attrs = set(g_mut.attributes())
    dataset = _norm(g_mut["dataset"]) if "dataset" in attrs else ""
    if dataset in {"cadets", "theia", "trace", "clearscope"}:
        return "darpa_e3"
    if dataset in {"cadets5", "theia5", "trace5", "clearscope5"}:
        return "darpa_e5"
    if dataset.startswith("optc"):
        return "optc"
    if dataset == "atlas":
        return "atlas"
    return "darpa_e3"


def imperceptibility_coverage(g_mut) -> Dict:
    """Classify every introduced/modified edge into user-visible categories."""
    family = _dataset_family(g_mut)
    counts = {category: 0 for category in PERCEIVABLE_ACTIONS[family]}
    matches = []
    audited_edges = 0
    for e_idx in range(g_mut.ecount()):
        e = g_mut.es[e_idx]
        edge_attrs = e.attributes()
        if not (
            bool(edge_attrs.get("_athena_boundary_edge", False))
            or bool(edge_attrs.get("_athena_edge_mutated", False))
            or bool(edge_attrs.get("_athena_introduced_edge", False))
        ):
            continue
        audited_edges += 1
        action = re.sub(r"[^a-z0-9]+", "_", _norm(edge_attrs.get("actions", ""))).strip("_")
        target_attrs = g_mut.vs[e.target].attributes()
        target_text = " ".join(
            _norm(target_attrs.get(key, ""))
            for key in ("type", "name", "properties", "path")
        )
        target_tokens = set(re.findall(r"[a-z0-9_./-]+", target_text))
        generic_action = action in {
            "create", "write", "read", "open", "show", "play",
            "event_create", "event_write", "event_read", "event_open",
            "file_create", "file_write", "file_read", "file_open",
        }
        for category, aliases in PERCEIVABLE_ACTIONS[family].items():
            markers = PERCEIVABLE_TARGET_MARKERS[category]
            marker_match = any(
                marker in target_tokens or marker in target_text for marker in markers
            )
            if action in aliases or (generic_action and marker_match):
                counts[category] += 1
                matches.append({
                    "edge": int(e_idx),
                    "event_id": str(edge_attrs.get("event_id", "")),
                    "action": action,
                    "category": category,
                    "source_type": _norm(g_mut.vs[e.source].attributes().get("type", "")),
                    "target_type": _norm(target_attrs.get("type", "")),
                })
    return {
        "dataset_family": family,
        "audited_edges": audited_edges,
        "category_counts": counts,
        "perceivable_matches": matches,
    }


def check_imperceptibility(g_mut, replaced_nodes: Set[int]) -> bool:
    """Check 3 (paper §IV.C) using explicit action/entity/target mappings."""
    del replaced_nodes
    return not imperceptibility_coverage(g_mut)["perceivable_matches"]


def check_hardness(g_mut, g_anchor, delta_h: float = 0.3) -> bool:
    """Check 4 (paper §IV.C). Rejects mutations whose WL
    similarity to the anchor is below ``delta_h``."""
    from .subgraph_retrieval import wl_kernel
    sim = wl_kernel(g_mut, g_anchor, h=3)
    return sim >= delta_h


def check_attack_chain_fidelity(g_mut) -> Tuple[bool, Dict]:
    """Verify every required attack-chain node and ordered event survived."""
    graph_attrs = set(g_mut.attributes())
    expected_nodes = (
        list(g_mut["_athena_required_attack_nodes"])
        if "_athena_required_attack_nodes" in graph_attrs else []
    )
    expected_edges = (
        list(g_mut["_athena_required_attack_edges"])
        if "_athena_required_attack_edges" in graph_attrs else []
    )
    introduced = any(
        bool(g_mut.vs[index].attributes().get("_athena_structurally_introduced", False))
        for index in range(g_mut.vcount())
    )
    if introduced and not expected_nodes:
        return False, {
            "required_nodes": 0, "preserved_nodes": 0,
            "required_edges": len(expected_edges), "preserved_edges": 0,
            "preserved_ratio": 0.0,
        }

    node_origins = {
        int(attrs.get("_athena_attack_source_vertex"))
        for attrs in (g_mut.vs[index].attributes() for index in range(g_mut.vcount()))
        if attrs.get("_athena_attack_source_vertex") is not None
    }
    preserved_nodes = sum(int(value) in node_origins for value in expected_nodes)
    observed_edges = set()
    for edge in g_mut.es:
        attrs = edge.attributes()
        if not bool(attrs.get("_athena_introduced_edge", False)):
            continue
        observed_edges.add((
            int(attrs.get("_athena_attack_source_vertex", -1)),
            int(attrs.get("_athena_attack_target_vertex", -1)),
            str(attrs.get("_athena_attack_source_event_id", "")),
            int(-1 if attrs.get("event_order") is None else attrs.get("event_order")),
            str(attrs.get("actions", "")),
        ))
    preserved_edges = 0
    for row in expected_edges:
        signature = (
            int(row["source_attack_vertex"]),
            int(row["target_attack_vertex"]),
            str(row["event_id"]),
            int(row["event_order"]),
            str(row["action"]),
        )
        preserved_edges += signature in observed_edges
    required_total = len(expected_nodes) + len(expected_edges)
    preserved_total = preserved_nodes + preserved_edges
    report = {
        "required_nodes": len(expected_nodes),
        "preserved_nodes": preserved_nodes,
        "required_edges": len(expected_edges),
        "preserved_edges": preserved_edges,
        "preserved_ratio": preserved_total / max(1, required_total),
    }
    return preserved_total == required_total, report


def collect_audited_nodes(g_mut, replaced_nodes: Set[int]) -> Set[int]:
    """Return the exact node set covered by unified verification."""
    audited_nodes = set(int(value) for value in replaced_nodes)
    if g_mut is None:
        return audited_nodes
    for vertex_index in range(g_mut.vcount()):
        attrs = g_mut.vs[vertex_index].attributes()
        if (
            bool(attrs.get("_athena_structurally_introduced", False))
            or bool(attrs.get("_athena_semantic_modified", False))
        ):
            audited_nodes.add(vertex_index)
    for edge in g_mut.es:
        attrs = edge.attributes()
        if any(bool(attrs.get(flag, False)) for flag in (
            "_athena_boundary_edge", "_athena_edge_mutated", "_athena_introduced_edge",
        )):
            audited_nodes.update((int(edge.source), int(edge.target)))
    return audited_nodes


def verify_mutation(
    g_mut,
    g_anchor,
    replaced_nodes: Set[int],
    entity_ops: Dict[str, Set[str]],
    type_attrs: Dict[str, Dict[str, Set[str]]],
    delta_h: float = 0.3,
) -> Tuple[bool, List[str]]:
    """Run all four checks. Returns ``(passed, failed_checks)``."""
    failed: List[str] = []

    audited_nodes = collect_audited_nodes(g_mut, replaced_nodes)

    if not check_operation_legality(g_mut, audited_nodes, entity_ops):
        failed.append("operation_legality")
    if not check_attribute_feasibility(g_mut, audited_nodes, type_attrs):
        failed.append("attribute_feasibility")
    if not check_imperceptibility(g_mut, audited_nodes):
        failed.append("imperceptibility")
    if g_mut is not None:
        fidelity_ok, _fidelity_report = check_attack_chain_fidelity(g_mut)
        if not fidelity_ok:
            failed.append("attack_chain_fidelity")
    if not check_hardness(g_mut, g_anchor, delta_h=delta_h):
        failed.append("hardness")

    return len(failed) == 0, failed
