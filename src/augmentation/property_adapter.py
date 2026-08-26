"""Lossless structured/legacy node-property access for augmentation."""
from __future__ import annotations

import ast
import json
from typing import Dict, List


def property_values(raw) -> List[str]:
    if isinstance(raw, (set, list, tuple)):
        return sorted({str(value).strip() for value in raw if str(value).strip()})
    text = str(raw or "").strip()
    if not text or text in {"set()", "{}"}:
        return []
    if text.startswith(("{", "[", "(")):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (set, list, tuple)):
            return sorted({str(value).strip() for value in parsed if str(value).strip()})
    return [text.strip("'\"")]


def _json_payload(raw):
    if not isinstance(raw, str):
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and isinstance(payload.get("events"), list):
        return payload
    return None


def process_records(raw) -> List[Dict[str, str]]:
    """Return every structured process observation without flattening JSON."""
    payload = _json_payload(raw)
    if payload is not None:
        records = []
        for event in payload["events"]:
            if not isinstance(event, dict):
                continue
            records.append({
                "event_id": str(event.get("event_id", "") or "").strip(),
                "event_order": str(event.get("event_order", "") or "").strip(),
                "role": str(event.get("role", "") or "").strip(),
                "action": str(event.get("action", "") or "").strip(),
                "command": str(event.get("command", "") or "").strip(),
                "arguments": str(event.get("arguments", "") or "").strip(),
                "tgid": "",
                "path": str(event.get("path", "") or "").strip(),
                "address": str(event.get("address", "") or "").strip(),
                "format": "atlas-json",
            })
        return records

    records = []
    for value in property_values(raw):
        parts = value.split(",", 2)
        command_line = parts[0].strip() if parts else ""
        command_parts = command_line.split(maxsplit=1)
        records.append({
            "event_id": "", "event_order": "", "role": "", "action": "",
            "command": command_parts[0] if command_parts else "",
            "arguments": command_parts[1] if len(command_parts) > 1 else "",
            "tgid": parts[1].strip() if len(parts) > 1 else "",
            "path": parts[2].strip() if len(parts) > 2 else "",
            "address": "",
            "format": "legacy",
        })
    return records


def process_view(raw) -> Dict[str, str]:
    payload = _json_payload(raw)
    if payload is not None:
        candidates = [
            (int(bool(row["command"])), int(bool(row["arguments"])),
             int(bool(row["path"])), -index, row)
            for index, row in enumerate(process_records(raw))
        ]
        if candidates:
            return max(candidates, key=lambda row: row[:-1])[-1]
        return {"command": "", "arguments": "", "tgid": "", "path": "", "address": "", "format": "atlas-json"}

    values = property_values(raw)
    candidates = []
    for value in values:
        parts = value.split(",", 2)
        command_line = parts[0].strip() if parts else ""
        command_parts = command_line.split(maxsplit=1)
        command = command_parts[0] if command_parts else ""
        arguments = command_parts[1] if len(command_parts) > 1 else ""
        tgid = parts[1].strip() if len(parts) > 1 else ""
        path = parts[2].strip() if len(parts) > 2 else ""
        candidates.append((
            int(bool(command) and "unknown-process" not in command),
            int(bool(path) and "unknown-path" not in path),
            value.count(","),
            len(value),
            value,
            {"command": command, "arguments": arguments, "tgid": tgid, "path": path, "address": "", "format": "legacy"},
        ))
    if not candidates:
        return {"command": "", "arguments": "", "tgid": "", "path": "", "address": "", "format": "legacy"}
    return max(candidates, key=lambda row: row[:-1])[-1]


def update_process_property(
    raw, old_command: str, old_arguments: str, new_command: str, new_arguments: str,
) -> str:
    payload = _json_payload(raw)
    if payload is not None:
        updated = json.loads(json.dumps(payload))
        changed = False
        for event in updated["events"]:
            if not isinstance(event, dict):
                continue
            command = str(event.get("command", "") or "").strip()
            arguments = str(event.get("arguments", "") or "").strip()
            if command == old_command and arguments == old_arguments:
                event["command"] = new_command
                event["arguments"] = new_arguments
                changed = True
        if not changed:
            raise ValueError("selected ATLAS process event disappeared before update")
        return json.dumps(updated, ensure_ascii=False, sort_keys=True)

    values = property_values(raw)
    view = process_view(raw)
    old_line = " ".join(part for part in (old_command, old_arguments) if part).strip()
    selected = f"{old_line},{view['tgid']},{view['path']}"
    new_line = " ".join(part for part in (new_command, new_arguments) if part).strip()
    replacement = f"{new_line},{view['tgid']},{view['path']}"
    if len(values) <= 1:
        return replacement
    replaced = False
    output = []
    for value in values:
        if not replaced and value == selected:
            output.append(replacement)
            replaced = True
        else:
            output.append(value)
    if not replaced:
        raise ValueError("selected legacy process property disappeared before update")
    return repr(sorted(output))


def typed_fields(raw, entity_type: str) -> Dict[str, set[str]]:
    vtype = str(entity_type or "").lower()
    if "process" in vtype or "subject" in vtype:
        records = process_records(raw)
        return {
            "properties.command": {row["command"].lower() for row in records if row["command"]},
            "properties.arguments": {row["arguments"].lower() for row in records if row["arguments"]},
            "properties.tgid": {row["tgid"].lower() for row in records if row["tgid"]},
            "properties.path": {row["path"].lower() for row in records if row["path"]},
            "properties.address": {row["address"].lower() for row in records if row["address"]},
        }
    payload = _json_payload(raw)
    if payload is not None:
        paths = {
            str(event.get("path", "") or "").strip().lower()
            for event in payload["events"] if isinstance(event, dict)
            if str(event.get("path", "") or "").strip()
        }
        addresses = {
            str(event.get("address", "") or "").strip().lower()
            for event in payload["events"] if isinstance(event, dict)
            if str(event.get("address", "") or "").strip()
        }
        if "file" in vtype:
            return {"properties.path": paths}
        if any(token in vtype for token in ("net", "flow", "sock")):
            return {"properties.address": addresses}
        return {"properties.path": paths, "properties.address": addresses}
    values = {value.lower() for value in property_values(raw) if value}
    if "file" in vtype:
        return {"properties.path": values}
    if any(token in vtype for token in ("net", "flow", "sock")):
        return {"properties.address": values}
    return {"properties": values}


def semantic_summary(raw, entity_type: str, max_chars: int = 192) -> str:
    """Bounded deterministic prompt/WL summary of type-relevant attributes."""
    fields = typed_fields(raw, entity_type)
    parts = []
    for field_name in sorted(fields):
        values = sorted(fields[field_name])
        if values:
            parts.append(f"{field_name.rsplit('.', 1)[-1]}={'|'.join(values)}")
    text = ";".join(parts) or "unknown"
    limit = max(24, int(max_chars))
    if len(text) <= limit:
        return text
    import hashlib
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{text[:limit - 17]}...#{digest}"


__all__ = [
    "process_records", "process_view", "property_values", "semantic_summary",
    "typed_fields", "update_process_property",
]
