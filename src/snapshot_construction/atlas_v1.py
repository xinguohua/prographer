"""Deterministic conversion of the official ATLAS v1 log release.

The upstream release stores each attack case below ``training_logs`` or
``testing_logs`` with a sibling ``malicious_labels.txt`` and a ``logs``
directory.  This module reads only those documented locations and converts
the raw Windows/Linux/DNS/Firefox records into ATHENA's event table.  It does
not crawl unrelated files; entity labels are projected separately from the
same-case official ``malicious_labels.txt`` using ATLAS's endpoint rule.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd


RAW_LOG_NAMES = ("security_events.txt", "audit.interpret.log", "dns", "firefox.txt")
EVENT_COLUMNS = (
    "event_id", "actorID", "actor_type", "objectID", "object", "action",
    "timestamp", "timestamp_unit", "command", "arguments", "path", "address",
    "src_address", "src_port", "dst_address", "dst_port",
)
_SCENARIO_RE = re.compile(r"(?<![A-Z0-9])([SM][1-6])(?![0-9])", re.IGNORECASE)
_HOST_RE = re.compile(r"(?<![A-Z0-9])(H\d+|EDR)(?![A-Z0-9])", re.IGNORECASE)


def normalize_atlas_value(value) -> str:
    """Apply the normalization used by the official ATLAS converter."""
    return str(value or "").strip().strip('"').replace("\\", "/").lower()


def parse_timezone_offset(value: str) -> timezone:
    """Parse an explicit ``Z``/``UTC``/``+HH:MM``/``-HH:MM`` offset."""
    raw = str(value or "").strip().upper()
    if raw in {"Z", "UTC", "+00:00", "-00:00"}:
        return timezone.utc
    match = re.fullmatch(r"([+-])(\d{2}):(\d{2})", raw)
    if not match:
        raise ValueError(
            "ATLAS source_timezone must be explicit: Z, UTC, +HH:MM, or -HH:MM"
        )
    hours, minutes = int(match.group(2)), int(match.group(3))
    if hours > 23 or minutes > 59:
        raise ValueError(f"invalid ATLAS source_timezone: {value!r}")
    delta = timedelta(hours=hours, minutes=minutes)
    if match.group(1) == "-":
        delta = -delta
    return timezone(delta)


def scenario_from_name(path: Path) -> str | None:
    for value in (path.name, *reversed(path.parts)):
        match = _SCENARIO_RE.search(str(value).replace("_", "-"))
        if match:
            return match.group(1).upper()
    return None


def host_from_name(path: Path) -> str:
    for value in (path.name, *reversed(path.parts)):
        match = _HOST_RE.search(str(value).replace("_", "-"))
        if match:
            return match.group(1).upper()
    lowered = path.name.lower()
    if "linux" in lowered:
        return "H1"
    if "windows" in lowered:
        return "H1"
    return "H1"


def discover_official_cases(root: Path, scenarios: Iterable[str]) -> list[Path]:
    """Return only documented ATLAS case directories for requested scenarios."""
    wanted = {str(value).upper() for value in scenarios}
    candidates: set[Path] = set()
    for split_name in ("training_logs", "testing_logs"):
        split_root = root / split_name
        if split_root.is_dir():
            candidates.update(path for path in split_root.iterdir() if path.is_dir())
    # The released per-scenario archives are also commonly unpacked one level
    # below a scenario directory.  Search exactly two documented levels, not
    # the entire user-supplied tree.
    for scenario in sorted(wanted):
        scenario_root = root / scenario
        if scenario_root.is_dir():
            candidates.add(scenario_root)
            candidates.update(path for path in scenario_root.iterdir() if path.is_dir())
            for split_name in ("training_logs", "testing_logs"):
                split_root = scenario_root / split_name
                if split_root.is_dir():
                    candidates.update(path for path in split_root.iterdir() if path.is_dir())
    return sorted(
        path for path in candidates
        if scenario_from_name(path) in wanted
        and (path / "logs").is_dir()
        and (path / "malicious_labels.txt").is_file()
    )


def discover_preprocessed_files(root: Path, scenarios: Iterable[str]) -> list[Path]:
    """Discover only official ATLAS preprocessed-output filenames."""
    wanted = {str(value).upper() for value in scenarios}
    roots = [root / "output", root / "paper_experiments" / "output", root / "paper_experiments"]
    for scenario in sorted(wanted):
        scenario_root = root / scenario
        roots.append(scenario_root / "output")
        if scenario_root.is_dir():
            host_roots = sorted(
                path for path in scenario_root.iterdir()
                if path.is_dir() and re.fullmatch(r"h\d+|edr", path.name, re.IGNORECASE)
            )
            # In the official multi-host archives, h1/output is the canonical
            # directory and already contains both the _h1 and _h2 files.  The
            # h2/output directory is a byte-for-byte duplicate.
            canonical_h1 = next(
                (path for path in host_roots if path.name.lower() == "h1" and (path / "output").is_dir()),
                None,
            )
            if scenario.startswith("M") and canonical_h1 is not None:
                roots.append(canonical_h1 / "output")
            else:
                roots.extend(path / "output" for path in host_roots)
    paths = []
    for directory in roots:
        if not directory.is_dir():
            continue
        for prefix in ("training_preprocessed_logs_", "testing_preprocessed_logs_"):
            paths.extend(path for path in directory.glob(prefix + "*") if path.is_file())
    candidates = sorted({
        path for path in paths
        if scenario_from_name(path) in wanted
        and not any(
            re.fullmatch(r"(?:h\d+[-_]h\d+|multi|_multi)", part, re.IGNORECASE)
            or "_multi" in part.lower()
            for part in path.parts
        )
    })
    # Every released fold is self-contained: a case appears once as the
    # testing stream in its own archive and again as a training stream in the
    # other folds.  M* archives additionally duplicate their h1/output files
    # below h2/output.  Prefix/path de-duplication is therefore insufficient.
    # Prefer the canonical testing copy, then de-duplicate by official case
    # name plus content hash so a case/host is ingested exactly once.
    candidates.sort(key=lambda path: (
        0 if path.name.startswith("testing_preprocessed_logs_") else 1,
        str(path),
    ))
    seen: set[tuple[str, str]] = set()
    unique: list[Path] = []
    for path in candidates:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        case_name = path.name
        for prefix in ("testing_preprocessed_logs_", "training_preprocessed_logs_"):
            if case_name.startswith(prefix):
                case_name = case_name[len(prefix):]
                break
        key = (case_name, digest.hexdigest())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_case_labels(case_dir: Path) -> set[str]:
    label_path = case_dir / "malicious_labels.txt"
    return {
        normalize_atlas_value(line)
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if normalize_atlas_value(line)
    }


def _event_id(case_dir: Path, source: str, ordinal: int, values: Iterable[str]) -> str:
    case_key = f"{case_dir.parent.name}/{case_dir.name}"
    digest = hashlib.sha256(
        "\x1f".join([case_key, source, str(ordinal), *map(str, values)]).encode("utf-8")
    ).hexdigest()
    return f"atlas-v1:{digest}"


def _parse_datetime(date_text: str, time_text: str, tz: timezone) -> float:
    date_text = date_text.strip()
    time_text = time_text.strip().split(".", 1)[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M:%S"):
        try:
            return datetime.strptime(f"{date_text} {time_text}", fmt).replace(tzinfo=tz).timestamp()
        except ValueError:
            pass
    raise ValueError(f"unsupported ATLAS timestamp: {date_text!r} {time_text!r}")


def _event(
    case_dir: Path,
    source: str,
    ordinal: int,
    timestamp: float,
    actor_id: str,
    actor_type: str,
    object_id: str,
    object_type: str,
    action: str,
    **fields,
) -> dict:
    actor_id = normalize_atlas_value(actor_id)
    object_id = normalize_atlas_value(object_id)
    if not actor_id or not object_id:
        raise ValueError("ATLAS event endpoints must be non-empty")
    row = {column: "" for column in EVENT_COLUMNS}
    row.update({
        "event_id": _event_id(case_dir, source, ordinal, (timestamp, actor_id, object_id, action)),
        "actorID": actor_id,
        "actor_type": actor_type,
        "objectID": object_id,
        "object": object_type,
        "action": normalize_atlas_value(action),
        "timestamp": float(timestamp),
        "timestamp_unit": "s",
    })
    for key, value in fields.items():
        if key in row:
            row[key] = normalize_atlas_value(value)
    return row


def normalize_file_event(process: str, file_name: str, raw_action: str) -> tuple[str, str, str, str, str]:
    """Canonicalize an ATLAS file event and its information-flow direction.

    Official raw and preprocessed inputs describe the same operations with
    different spellings.  Reads and execution move information from the file
    to the process; writes and deletion move from the process to the file.
    """
    value = normalize_atlas_value(raw_action)
    compact = re.sub(r"[^a-z0-9]+", "", value)
    if any(token in compact for token in ("delete", "remove", "unlink")):
        action, file_to_process = "file_delete", False
    elif any(token in compact for token in ("write", "append", "truncate", "rename")):
        action, file_to_process = "file_write", False
    elif any(token in compact for token in ("execute", "execve", "exec")):
        action, file_to_process = "file_execute", True
    elif any(token in compact for token in ("readdata", "read", "open")):
        action, file_to_process = "file_read", True
    else:
        action, file_to_process = "file_access", False
    process = normalize_atlas_value(process)
    file_name = normalize_atlas_value(file_name)
    if file_to_process:
        return file_name, "file", process, "process", action
    return process, "process", file_name, "file", action


def _windows_blocks(path: Path) -> Iterator[tuple[str, list[str]]]:
    header = ""
    body: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.rstrip()
        if line.startswith(("Information ", "Audit Success ", "Audit Failure ")):
            if header:
                yield header, body
            header, body = line, []
        elif header:
            body.append(line)
    if header:
        yield header, body


def _kv_value(lines: list[str], *names: str) -> str:
    for line in lines:
        stripped = line.strip()
        for name in names:
            if stripped.lower().startswith(name.lower() + ":"):
                return stripped.split(":", 1)[1].strip().strip('"')
    return ""


def _parse_windows(case_dir: Path, path: Path, tz: timezone) -> Iterator[dict]:
    process_names: dict[str, str] = {}
    ordinal = 0
    for header, body in _windows_blocks(path):
        match = re.search(
            r"(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+(?P<time>\d{1,2}:\d{2}:\d{2})\s+(?P<ampm>AM|PM)",
            header,
            re.IGNORECASE,
        )
        if not match:
            continue
        time_text = match.group("time") + " " + match.group("ampm").upper()
        parsed = datetime.strptime(
            f"{match.group('date')} {time_text}", "%m/%d/%Y %I:%M:%S %p"
        ).replace(tzinfo=tz).timestamp()
        pid = _kv_value(body, "New Process ID", "Process ID")
        ppid = _kv_value(body, "Creator Process ID")
        pname = normalize_atlas_value(
            _kv_value(body, "New Process Name", "Process Name", "Application Name")
        )
        if pid.startswith("0x"):
            try:
                pid = str(int(pid, 16))
            except ValueError:
                pass
        if ppid.startswith("0x"):
            try:
                ppid = str(int(ppid, 16))
            except ValueError:
                pass
        if pid and pname:
            process_names[pid] = pname
        process = f"{pname or 'noprocessname'}_{pid or 'nopid'}"
        if ppid:
            parent = f"{process_names.get(ppid, 'noprocessname')}_{ppid}"
            ordinal += 1
            yield _event(
                case_dir, path.name, ordinal, parsed, process, "process", parent, "process",
                "create_process", command=pname, arguments="", path=pname,
            )
        object_name = _kv_value(body, "Object Name")
        accesses = _kv_value(body, "Accesses") or _kv_value(body, "Access Mask")
        if object_name:
            actor, actor_type, obj, object_type, action = normalize_file_event(
                process, object_name, accesses or "access_file"
            )
            ordinal += 1
            yield _event(
                case_dir, path.name, ordinal, parsed, actor, actor_type, obj, object_type,
                action, command=pname, path=object_name,
            )
        src_ip = _kv_value(body, "Source Address")
        src_port = _kv_value(body, "Source Port")
        dst_ip = _kv_value(body, "Destination Address")
        dst_port = _kv_value(body, "Destination Port")
        if dst_ip:
            endpoint = f"{dst_ip}:{dst_port}" if dst_port else dst_ip
            ordinal += 1
            yield _event(
                case_dir, path.name, ordinal, parsed, process, "process", endpoint, "network",
                _kv_value(body, "Direction") or "network_connect", command=pname,
                address=endpoint, src_address=src_ip, src_port=src_port,
                dst_address=dst_ip, dst_port=dst_port,
            )


def _parse_linux(case_dir: Path, path: Path, tz: timezone) -> Iterator[dict]:
    records = path.read_text(encoding="utf-8", errors="ignore").split("----")
    ordinal = 0
    for record in records:
        entry = " ".join(line.strip().strip("'") for line in record.splitlines())
        date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", entry)
        time_match = re.search(r"(\d{1,2}:\d{2}:\d{2})(?:\.\d+)?", entry)
        if not date_match or not time_match:
            continue
        timestamp = _parse_datetime(date_match.group(1), time_match.group(1), tz)
        def field(name: str) -> str:
            match = re.search(rf"(?:^|\s){re.escape(name)}=(?:\"([^\"]*)\"|'([^']*)'|(\S+))", entry)
            return next((value for value in match.groups() if value is not None), "") if match else ""
        pid, ppid = field("pid"), field("ppid")
        exe = normalize_atlas_value(field("exe") or field("comm") or "noprocessname")
        process = f"{exe}_{pid or 'nopid'}"
        if ppid:
            ordinal += 1
            yield _event(
                case_dir, path.name, ordinal, timestamp, process, "process",
                f"noprocessname_{ppid}", "process", "create_process",
                command=exe, arguments=field("proctitle"), path=exe,
            )
        syscall = field("syscall") or field("type") or "audit_event"
        object_name = field("name") or (field("a0") if any(x in syscall.lower() for x in ("exec", "remove")) else "")
        if object_name:
            actor, actor_type, obj, object_type, action = normalize_file_event(
                process, object_name, syscall
            )
            ordinal += 1
            yield _event(
                case_dir, path.name, ordinal, timestamp, actor, actor_type, obj, object_type,
                action, command=exe, arguments=field("proctitle"), path=object_name,
            )
        dst_ip, dst_port = field("host"), field("serv")
        if dst_ip:
            endpoint = f"{dst_ip}:{dst_port}" if dst_port else dst_ip
            ordinal += 1
            yield _event(
                case_dir, path.name, ordinal, timestamp, process, "process", endpoint, "network",
                syscall, command=exe, arguments=field("proctitle"), address=endpoint,
                dst_address=dst_ip, dst_port=dst_port,
            )


def _parse_dns(case_dir: Path, path: Path, tz: timezone) -> Iterator[dict]:
    ordinal = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "response" not in line.lower():
            continue
        parts = line.split()
        if len(parts) < 9:
            continue
        timestamp = _parse_datetime(parts[1], parts[2], timezone.utc)
        info = " ".join(parts[8:])
        domain_match = re.search(r"response\s+\S+\s+A+\s+(\S+)", info, re.IGNORECASE)
        ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", info)
        if not domain_match or not ip_match:
            continue
        ordinal += 1
        yield _event(
            case_dir, path.name, ordinal, timestamp, domain_match.group(1), "domain",
            ip_match.group(0), "network", "resolve", address=ip_match.group(0),
            src_address=parts[3], dst_address=parts[5],
        )


def _parse_firefox(case_dir: Path, path: Path, tz: timezone) -> Iterator[dict]:
    # Firefox timestamps are UTC in the release; converting from UTC preserves
    # the same instant as the upstream fixed -5 hour adjustment.
    ordinal = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        dt_match = re.search(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})(?:\.\d+)?", line)
        url_match = re.search(r"https?://([^\s\]\"']+)", line, re.IGNORECASE)
        if not dt_match or not url_match:
            continue
        timestamp = _parse_datetime(dt_match.group(1), dt_match.group(2), timezone.utc)
        url = normalize_atlas_value(url_match.group(0))
        host = normalize_atlas_value(url_match.group(1).split("/", 1)[0].split(":", 1)[0])
        ordinal += 1
        yield _event(
            case_dir, path.name, ordinal, timestamp, url, "web_object", host, "domain",
            "web_request", path=url, address=host,
        )


def convert_official_case(case_dir: Path, source_timezone: str) -> pd.DataFrame:
    """Convert one official case directory to the canonical event schema."""
    tz = parse_timezone_offset(source_timezone)
    logs = case_dir / "logs"
    parsers = {
        "security_events.txt": _parse_windows,
        "audit.interpret.log": _parse_linux,
        "dns": _parse_dns,
        "firefox.txt": _parse_firefox,
    }
    rows: list[dict] = []
    for name in RAW_LOG_NAMES:
        path = logs / name
        if path.is_file():
            rows.extend(parsers[name](case_dir, path, tz))
    if not rows:
        raise RuntimeError(f"ATLAS official case has no convertible events: {case_dir}")
    frame = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    frame["_atlas_case"] = case_dir.name
    return frame.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)


def convert_preprocessed_file(path: Path) -> pd.DataFrame:
    """Convert an official 20-field ``*_preprocessed_logs_*`` file."""
    rows: list[dict] = []
    processes: dict[str, str] = {}
    ordinal = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.search(r"-(LA|LB|LD)([+-])$", line, re.IGNORECASE)
        if not match:
            raise ValueError(f"ATLAS preprocessed line lacks -LA/-LB/-LD label: {path}")
        malicious = match.group(2) == "+"
        payload = line[:match.start()]
        fields = payload.split(",")
        # The official S2/S3/S4 Firefox export contains a documented 19-field
        # LB row shape where the host-domain column (index 14) is omitted. This
        # is the only structural repair admitted; every other width is invalid.
        if len(fields) == 19 and match.group(1).upper() == "LB":
            fields.insert(14, "")
        if len(fields) != 20:
            raise ValueError(f"ATLAS preprocessed line must have 20 fields: {path}")
        try:
            timestamp = float(fields[0])
        except ValueError as exc:
            raise ValueError(f"invalid ATLAS preprocessed timestamp: {fields[0]!r}") from exc

        def append(actor, actor_type, obj, object_type, action, **semantics):
            nonlocal ordinal
            ordinal += 1
            event = _event(
                path.parent,
                path.name,
                ordinal,
                timestamp,
                actor,
                actor_type,
                obj,
                object_type,
                action,
                **semantics,
            )
            event["malicious_event"] = bool(malicious)
            rows.append(event)

        domain, resolved_ip = fields[1], fields[2]
        if domain and resolved_ip:
            append(domain, "domain", resolved_ip, "network", "resolve", address=resolved_ip)

        pid, ppid, pname = fields[3], fields[4], fields[5]
        process = ""
        if pid:
            if pname:
                processes[pid] = pname
            process = f"{pname or 'noprocessname'}_{pid}"
            if ppid:
                parent = f"{processes.get(ppid, 'noprocessname')}_{ppid}"
                append(process, "process", parent, "process", "create_process", command=pname, path=pname)

        src_ip, src_port, dst_ip, dst_port = fields[6], fields[7], fields[8], fields[9]
        if process and dst_ip:
            endpoint = f"{dst_ip}:{dst_port}" if dst_port else dst_ip
            append(
                process,
                "process",
                endpoint,
                "network",
                "network_connect",
                command=pname,
                address=endpoint,
                src_address=src_ip,
                src_port=src_port,
                dst_address=dst_ip,
                dst_port=dst_port,
            )

        url = fields[11] or fields[12] or fields[16]
        host = fields[14]
        if url and host:
            append(url, "web_object", host, "domain", fields[10] or "web_request", path=url, address=host)

        accesses, object_name = fields[17], fields[18]
        if process and accesses.startswith("file_") and object_name:
            actor, actor_type, obj, object_type, action = normalize_file_event(
                process, object_name, accesses
            )
            append(
                actor, actor_type, obj, object_type, action,
                command=pname, path=object_name,
            )
    if not rows:
        raise RuntimeError(f"ATLAS preprocessed file contains no graph events: {path}")
    frame = pd.DataFrame(rows)
    frame["_atlas_case"] = path.name
    return frame.sort_values(["timestamp", "event_id"], kind="stable").reset_index(drop=True)


def resolve_case_label_ids(
    case_dir: Path,
    source_labels: set[str],
    frame: pd.DataFrame,
) -> tuple[set[str], dict]:
    """Project official labels endpoint-wise using ATLAS's substring rule."""
    resolved: set[str] = set()
    matched_labels: set[str] = set()
    event_matches = []
    normalized_labels = {normalize_atlas_value(label) for label in source_labels if label}
    for row in frame.itertuples(index=False):
        actor = normalize_atlas_value(row.actorID)
        obj = normalize_atlas_value(row.objectID)
        actor_matches = sorted(label for label in normalized_labels if label in actor)
        object_matches = sorted(label for label in normalized_labels if label in obj)
        if actor_matches:
            resolved.add(actor)
        if object_matches:
            resolved.add(obj)
        row_matches = sorted(set(actor_matches + object_matches))
        matched_labels.update(row_matches)
        event_matches.append(row_matches)
    frame["matched_labels"] = [json.dumps(values) for values in event_matches]
    frame["label_source"] = str(case_dir / "malicious_labels.txt")
    if "malicious_event" not in frame.columns:
        frame["malicious_event"] = [bool(values) for values in event_matches]
    audit = {
        "case": case_dir.name,
        "source_label_count": len(source_labels),
        "matched_source_label_count": len(matched_labels),
        "resolved_entity_count": len(resolved),
        "coverage": len(matched_labels) / max(1, len(source_labels)),
        "label_source": str(case_dir / "malicious_labels.txt"),
    }
    if source_labels and not matched_labels:
        raise RuntimeError(
            f"official ATLAS labels for {case_dir.name} match no converted graph endpoint"
        )
    return resolved, audit


__all__ = [
    "EVENT_COLUMNS", "RAW_LOG_NAMES", "convert_official_case", "convert_preprocessed_file",
    "discover_official_cases", "discover_preprocessed_files",
    "host_from_name", "load_case_labels", "normalize_atlas_value", "parse_timezone_offset",
    "resolve_case_label_ids", "scenario_from_name",
]
