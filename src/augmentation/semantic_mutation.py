"""Paper Section IV-C2 semantic mutation.

Each injected process node is assigned exactly one of Replacement, Rewriting,
or Extension from the historical benign corpus ``H_b``. The strategy-specific
released prompt is executed and validated before graph attributes are changed.
"""
from __future__ import annotations

import json
import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .property_adapter import process_records, process_view, property_values, update_process_property


PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
STRATEGY_PROMPTS = {
    "replacement": PROMPT_DIR / "replacement.txt",
    "rewriting": PROMPT_DIR / "rewriting.txt",
    "extension": PROMPT_DIR / "extension.txt",
}


def _property_values(prop) -> List[str]:
    return property_values(prop)


def _select_property(values: List[str], entity_type: str) -> str:
    if not values:
        return ""
    vtype = str(entity_type).lower()
    if "process" in vtype or "subject" in vtype:
        def score(value: str):
            command, _args, _tgid, path = _command_components(value)
            return (
                int(bool(command) and "unknown-process" not in command),
                int(bool(path) and "unknown-path" not in path),
                value.count(","),
                len(value),
                value,
            )
        return max(values, key=score)
    if "file" in vtype:
        return max(values, key=lambda value: (int("unknown-file" not in value), len(value), value))
    if any(token in vtype for token in ("net", "flow", "sock")):
        return max(values, key=lambda value: (int("snapshot-local-network" not in value), len(value), value))
    return max(values, key=lambda value: (len(value), value))


def _get_properties(graph, vertex_index: int) -> str:
    try:
        raw = graph.vs[vertex_index]["properties"]
    except Exception:
        raw = graph.vs[vertex_index].attributes().get("properties", "")
    entity_type = graph.vs[vertex_index].attributes().get("type", "")
    if "process" in str(entity_type).lower() or "subject" in str(entity_type).lower():
        view = process_view(raw)
        command_line = " ".join(
            part for part in (view["command"], view["arguments"]) if part
        ).strip()
        return f"{command_line},{view['tgid']},{view['path']}"
    return _select_property(_property_values(raw), str(entity_type))


def _parse_process_properties(prop: str) -> Tuple[str, str, str]:
    view = process_view(prop)
    command_line = " ".join(part for part in (view["command"], view["arguments"]) if part).strip()
    tgid = view["tgid"]
    path = view["path"]
    return command_line, tgid, path


def _command_components(prop: str) -> Tuple[str, str, str, str]:
    command_line, tgid, path = _parse_process_properties(prop)
    parts = command_line.split(maxsplit=1)
    command_name = parts[0] if parts else ""
    arguments = parts[1] if len(parts) > 1 else ""
    return command_name, arguments, tgid, path


def _strip_enclosing_quotes(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _shell_body(command: str, arguments: str) -> Optional[str]:
    if command not in {"sh", "bash", "dash", "zsh"}:
        return None
    match = re.fullmatch(r"-c\s+(.+)", arguments, flags=re.S)
    if not match:
        return None
    return _strip_enclosing_quotes(match.group(1)).replace(r'\"', '"').replace(r"\'", "'")


def _collect_benign_corpus(benign_graphs: list) -> Tuple[Set[str], Set[str], Set[str]]:
    """Build ``H_b`` command-name, complete-argument, and file-path sets."""
    commands: Set[str] = set()
    arguments: Set[str] = set()
    files: Set[str] = set()
    for graph, _reference in benign_graphs:
        for vertex_index in range(graph.vcount()):
            attrs = graph.vs[vertex_index].attributes()
            if int(attrs.get("label", 0) or 0) != 0:
                continue
            entity_type = str(attrs.get("type", "")).lower()
            prop = _get_properties(graph, vertex_index)
            if "process" in entity_type or "subject" in entity_type:
                for row in process_records(attrs.get("properties", "")):
                    if row["command"]:
                        commands.add(row["command"])
                        arguments.add(row["arguments"])
            elif "file" in entity_type and prop:
                files.add(prop)
    return commands, arguments, files


def _assign_strategy(
    prop: str,
    vtype: str,
    benign_commands: Set[str],
    benign_args: Set[str],
) -> str:
    """Assign one mutually exclusive strategy from membership in ``H_b``."""
    if "process" not in vtype and "subject" not in vtype:
        raise ValueError("semantic mutation is defined for process nodes")
    command_name, arguments, _tgid, _path = _command_components(prop)
    if not command_name:
        raise ValueError("process node has no command name")
    command_seen = command_name in benign_commands
    arguments_seen = arguments in benign_args
    if command_seen != arguments_seen:
        return "replacement"
    if command_seen:
        return "rewriting"
    return "extension"


def _build_context_triples(graph, node_index: int, r_hop: int = 2) -> List[str]:
    del r_hop  # C membership was already bounded during structural replacement.
    triples: List[str] = []
    allowed = {int(node_index)} | {
        index for index in range(graph.vcount())
        if bool(graph.vs[index].attributes().get("_athena_boundary_context", False))
    }
    for edge_index in range(graph.ecount()):
        edge = graph.es[edge_index]
        if edge.source not in allowed or edge.target not in allowed:
            continue
        if edge.source != node_index and edge.target != node_index and not (
            bool(graph.vs[edge.source].attributes().get("_athena_boundary_context", False))
            and bool(graph.vs[edge.target].attributes().get("_athena_boundary_context", False))
        ):
            continue
        action = str(edge.attributes().get("actions", "UNK"))
        src_type = str(graph.vs[edge.source].attributes().get("type", "UNK"))
        dst_type = str(graph.vs[edge.target].attributes().get("type", "UNK"))
        event_id = str(edge.attributes().get("event_id", "") or "")

        def summaries(vertex_index: int, role: str) -> List[str]:
            attrs = graph.vs[vertex_index].attributes()
            entity_type = str(attrs.get("type", ""))
            records = process_records(attrs.get("properties", ""))
            relevant = [
                row for row in records
                if (not event_id or not row.get("event_id") or row.get("event_id") == event_id)
                and (not row.get("role") or row.get("role") == role)
                and (not row.get("action") or row.get("action") == action)
            ]
            if relevant and ("process" in entity_type.lower() or "subject" in entity_type.lower()):
                return [
                    " ".join(part for part in (
                        row["command"], row["arguments"], row["path"], row["address"],
                    ) if part)[:192]
                    for row in relevant
                ]
            return [_get_properties(graph, vertex_index)[:192]]

        for src_prop in summaries(edge.source, "actor"):
            for dst_prop in summaries(edge.target, "object"):
                triples.append(f"<{src_type}:{src_prop}, {action}, {dst_type}:{dst_prop}>")
    return triples


def _context_benign_components(
    graph, node_index: int, r_hop: int,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Collect process components actually present in this node's benign C."""
    commands: Set[str] = set()
    arguments: Set[str] = set()
    command_lines: Set[str] = set()
    for neighbor in graph.neighborhood(node_index, order=int(r_hop), mode="all"):
        if int(neighbor) == int(node_index):
            continue
        attrs = graph.vs[int(neighbor)].attributes()
        if not bool(attrs.get("_athena_boundary_context", False)):
            continue
        if int(attrs.get("label", 0) or 0) != 0:
            continue
        entity_type = str(attrs.get("type", "")).lower()
        if "process" not in entity_type and "subject" not in entity_type:
            continue
        for row in process_records(attrs.get("properties", "")):
            command, args = row["command"], row["arguments"]
            if command:
                commands.add(command)
                command_lines.add(" ".join(part for part in (command, args) if part).strip())
                body = _shell_body(command, args)
                if body:
                    command_lines.add(body)
            if args:
                arguments.add(args)
    return commands, arguments, command_lines


def _render_prompt(
    strategy: str,
    command_name: str,
    arguments: str,
    context: List[str],
    preserve: Optional[str] = None,
    max_prompt_chars: int = 8800,
) -> Tuple[str, Dict[str, int | bool]]:
    template = STRATEGY_PROMPTS[strategy].read_text(encoding="utf-8")
    template = template.replace("{name}", command_name).replace("{args}", arguments)
    details = [
        "",
        "[Concrete Input]",
        f'command_name = "{command_name}"',
        f'arguments = "{arguments}"',
    ]
    if preserve is not None:
        details.append(f'preserve = "{preserve}"')
    details.append("Context C:")
    suffix = "Return only the JSON object specified above."
    used: List[str] = []
    for row in context:
        candidate = template + "\n" + "\n".join(details + used + [row, suffix])
        if len(candidate) > int(max_prompt_chars):
            break
        used.append(row)
    prompt = template + "\n" + "\n".join(details + (used or ["[]"]) + [suffix])
    if len(prompt) > int(max_prompt_chars):
        raise ValueError("semantic mutation base prompt exceeds configured character budget")
    return prompt, {
        "context_triples_total": len(context),
        "context_triples_used": len(used),
        "context_truncated": len(used) < len(context),
        "prompt_chars": len(prompt),
        "prompt_char_budget": int(max_prompt_chars),
    }


def _parse_llm_response(response: str) -> Dict[str, str]:
    text = str(response).strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("semantic mutation response is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("semantic mutation response must be one JSON object")
    command_name = result.get("new_command_name")
    arguments = result.get("new_arguments")
    if not isinstance(command_name, str) or not command_name.strip():
        raise ValueError("semantic mutation response lacks new_command_name")
    if not isinstance(arguments, str):
        raise ValueError("semantic mutation response lacks new_arguments")
    return {
        "new_command_name": command_name.strip(),
        "new_arguments": arguments.strip(),
    }


def _validate_output(
    strategy: str,
    old_command: str,
    old_arguments: str,
    new_command: str,
    new_arguments: str,
    command_seen: bool,
    arguments_seen: bool,
    benign_commands: Set[str],
    benign_args: Set[str],
    context_commands: Set[str],
    context_args: Set[str],
    context_command_lines: Set[str],
) -> Set[str]:
    old_full = " ".join(part for part in (old_command, old_arguments) if part).strip()
    new_full = " ".join(part for part in (new_command, new_arguments) if part).strip()
    changed = {
        field for field, old, new in (
            ("command_name", old_command, new_command),
            ("arguments", old_arguments, new_arguments),
        ) if old != new
    }
    if not changed:
        raise ValueError(f"{strategy} returned a no-op")
    if strategy == "replacement":
        if command_seen:
            starts = [
                match.start() for match in re.finditer(re.escape(old_arguments), new_arguments)
                if (match.start() == 0 or new_arguments[match.start() - 1].isspace())
                and (match.end() == len(new_arguments) or new_arguments[match.end()].isspace())
            ] if old_arguments else [0]
            if not starts:
                raise ValueError("replacement lost or reordered attack-specific arguments")
            match_start = starts[0]
            extras = [
                new_arguments[:match_start].strip(),
                new_arguments[match_start + len(old_arguments):].strip(),
            ]
            if any(extra and extra not in context_args and extra not in benign_args for extra in extras):
                raise ValueError("replacement added arguments not observed in benign C/H_b")
            if new_command == old_command or new_command not in benign_commands:
                raise ValueError("replacement command must be a different value observed in H_b")
        else:
            if new_command != old_command:
                raise ValueError("replacement changed the attack-specific command name")
            if changed != {"arguments"} or new_arguments not in benign_args:
                raise ValueError("replacement arguments must be a different value observed in H_b")
    elif strategy == "rewriting":
        if new_command not in benign_commands or new_arguments not in benign_args:
            raise ValueError("rewriting values must be observed in H_b")
    elif strategy == "extension":
        extended = new_full
        if new_command in {"sh", "bash", "dash", "zsh"}:
            extended = _shell_body(new_command, new_arguments)
            if extended is None:
                raise ValueError("extension shell wrapper must use a complete -c command body")
        if old_full not in extended:
            raise ValueError("extension did not preserve the complete attack command")
        before, after = extended.split(old_full, 1)
        segments = [
            _strip_enclosing_quotes(segment)
            for segment in re.split(r"&&|;", before + " ; " + after)
            if _strip_enclosing_quotes(segment)
        ]
        if not segments or any(segment not in context_command_lines for segment in segments):
            raise ValueError("extension prefix/suffix must preserve a benign command/argument sequence from C")
    else:
        raise ValueError(f"unknown semantic mutation strategy: {strategy}")
    return changed


def _propagate_associated_attributes(
    graph,
    process_index: int,
    old_command: str,
    old_arguments: str,
    new_command: str,
    new_arguments: str,
) -> None:
    """Mechanically propagate exact command/argument substitutions to neighbors."""
    substitutions = []
    if old_command != new_command:
        substitutions.append((old_command, new_command))
    if old_arguments and old_arguments != new_arguments:
        substitutions.append((old_arguments, new_arguments))
    modified: Set[int] = set()
    for neighbor in graph.neighbors(process_index, mode="all"):
        entity_type = str(graph.vs[neighbor].attributes().get("type", "")).lower()
        if not any(token in entity_type for token in ("file", "net", "flow", "sock")):
            continue
        prop = _get_properties(graph, neighbor)
        updated = prop
        for old, new in substitutions:
            if updated == old:
                updated = new
                continue
            pattern = rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])"
            updated = re.sub(pattern, new, updated)
        if updated != prop:
            neighbor_attrs = graph.vs[neighbor].attributes()
            if not neighbor_attrs.get("_athena_attack_original_mutable_attributes"):
                graph.vs[neighbor]["_athena_attack_original_mutable_attributes"] = {
                    "properties": prop
                }
            graph.vs[neighbor]["properties"] = updated
            graph.vs[neighbor]["_athena_semantic_modified"] = True
            graph.vs[neighbor]["_athena_semantic_changed_fields"] = ["properties"]
            graph.vs[neighbor]["_athena_semantic_before"] = prop
            graph.vs[neighbor]["_athena_semantic_after"] = updated
            modified.add(int(neighbor))
    return modified


def apply_semantic_mutation_llm(
    g_mut,
    attack_node_indices: List[int],
    benign_commands: Set[str],
    benign_args: Set[str],
    llm_fn=None,
    r_hop: int = 2,
    model_name: str = "unknown",
    max_prompt_chars: int = 8800,
):
    """Apply one validated strategy per injected process node, or return ``None``."""
    del model_name  # recorded by the augmentation manifest
    if llm_fn is None:
        return None

    planned = []
    process_nodes = []
    for node_index in sorted(set(int(value) for value in attack_node_indices)):
        if node_index < 0 or node_index >= g_mut.vcount():
            return None
        entity_type = str(g_mut.vs[node_index].attributes().get("type", "")).lower()
        if "process" not in entity_type and "subject" not in entity_type:
            continue
        process_nodes.append(node_index)
    try:
        signature = inspect.signature(llm_fn)
        accepts_metadata = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_metadata = False
    for batch_index, node_index in enumerate(process_nodes, 1):
        entity_type = str(g_mut.vs[node_index].attributes().get("type", "")).lower()
        prop = _get_properties(g_mut, node_index)
        raw_prop = g_mut.vs[node_index].attributes().get("properties", "")
        old_command, old_arguments, tgid, path = _command_components(prop)
        try:
            strategy = _assign_strategy(prop, entity_type, benign_commands, benign_args)
            command_seen = old_command in benign_commands
            arguments_seen = old_arguments in benign_args
            context_commands, context_args, context_command_lines = _context_benign_components(
                g_mut, node_index, r_hop,
            )
            preserve = None
            if strategy == "replacement":
                preserve = "arguments" if command_seen else "command_name"
            prompt, prompt_metadata = _render_prompt(
                strategy,
                old_command,
                old_arguments,
                _build_context_triples(g_mut, node_index, r_hop=r_hop),
                preserve=preserve,
                max_prompt_chars=max_prompt_chars,
            )
            prompt_metadata.update({
                "batch_index": batch_index,
                "batch_count": len(process_nodes),
                "candidate_count": 1,
            })
            raw_response = (
                llm_fn(prompt, **prompt_metadata) if accepts_metadata else llm_fn(prompt)
            )
            result = _parse_llm_response(raw_response)
            new_command = result["new_command_name"]
            new_arguments = result["new_arguments"]
            changed_fields = _validate_output(
                strategy,
                old_command,
                old_arguments,
                new_command,
                new_arguments,
                command_seen,
                arguments_seen,
                benign_commands,
                benign_args,
                context_commands,
                context_args,
                context_command_lines,
            )
        except Exception as exc:
            print(f"[SemMut] rejected node {node_index}: {exc}")
            return None
        new_line = " ".join(part for part in (new_command, new_arguments) if part).strip()
        new_prop = update_process_property(
            raw_prop,
            old_command,
            old_arguments,
            new_command,
            new_arguments,
        )
        planned.append({
            "node_index": node_index,
            "strategy": strategy,
            "old_command": old_command,
            "old_arguments": old_arguments,
            "new_command": new_command,
            "new_arguments": new_arguments,
            "before": " ".join(part for part in (old_command, old_arguments) if part).strip(),
            "after": new_line,
            "changed_fields": sorted(changed_fields),
            "new_prop": new_prop,
            "raw_before": raw_prop,
        })

    if not planned:
        return g_mut
    for mutation in planned:
        node_index = mutation["node_index"]
        attrs = g_mut.vs[node_index].attributes()
        if not attrs.get("_athena_attack_original_mutable_attributes"):
            g_mut.vs[node_index]["_athena_attack_original_mutable_attributes"] = {
                "properties": mutation["raw_before"]
            }
        g_mut.vs[node_index]["properties"] = mutation["new_prop"]
        g_mut.vs[node_index]["_athena_semantic_modified"] = True
        g_mut.vs[node_index]["_athena_semantic_strategy"] = mutation["strategy"]
        g_mut.vs[node_index]["_athena_semantic_before"] = mutation["before"]
        g_mut.vs[node_index]["_athena_semantic_after"] = mutation["after"]
        g_mut.vs[node_index]["_athena_semantic_changed_fields"] = mutation["changed_fields"]
    for mutation in planned:
        _propagate_associated_attributes(
            g_mut,
            mutation["node_index"],
            mutation["old_command"],
            mutation["old_arguments"],
            mutation["new_command"],
            mutation["new_arguments"],
        )
    return g_mut
