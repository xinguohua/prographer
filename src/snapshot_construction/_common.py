import csv
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]

_RELEASED_LABEL_DIR = {
    "cadets": "darpa_e3",
    "theia": "darpa_e3",
    "trace": "darpa_e3",
    "clearscope": "darpa_e3",
    "cadets5": "darpa_e5",
    "theia5": "darpa_e5",
    "trace5": "darpa_e5",
    "clearscope5": "darpa_e5",
    "optcday1": "optc",
}

_RELEASED_LABEL_PREFIX = {
    "cadets": "cadets",
    "theia": "theia",
    "trace": "trace",
    "clearscope": "clearscope",
    "cadets5": "cadets",
    "theia5": "theia",
    "trace5": "trace",
    "clearscope5": "clearscope",
}

# PIDSMaker publishes E5 malicious provenance-node UUIDs by attack, not by
# ATHENA's local indexed-scene filenames.  Keep the exact source files for
# each audit platform explicit so a scene filter cannot silently fall back to
# an unrelated legacy label file or union labels from another platform.
_E5_PIDSMaker_LABEL_FILES = {
    "cadets5": (
        "cadets_nginx_drakon_apt.txt",
        "cadets_nginx_drakon_apt_17.txt",
    ),
    "theia5": (
        "theia_firefox_drakon_apt_binfmt_elevate_inject.txt",
    ),
    "trace5": (
        "trace_firefox_drakon.txt",
    ),
    "clearscope5": (
        "clearscope_appstarter_0515.txt",
        "clearscope_firefox_0517.txt",
        "clearscope_lockwatch_0517.txt",
        "clearscope_tester_0517.txt",
    ),
}

_OPTC_PAPER_LABEL_FILES = ("host_0051.txt", "host_0201.txt", "host_0501.txt")
_OPTC_PAPER_LABEL_BY_HOST = {
    "H051": "host_0051.txt",
    "H201": "host_0201.txt",
    "H501": "host_0501.txt",
}


def collect_dot_paths(base_dir):
    result = []
    for file in os.listdir(base_dir):
        full_path = os.path.join(base_dir, file)
        if os.path.isfile(full_path) and file.endswith(".dot"):  # iffileis .dot 
            result.append(full_path)  # use append addfilepathtolist
    return result  # return .dot filepath's list

def collect_json_paths(base_dir):
    result = {}
    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            result[subdir] = {"benign": [], "malicious": []}
            for category in ["benign", "malicious"]:
                category_path = os.path.join(subdir_path, category)
                if os.path.exists(category_path):
                    for file in sorted(os.listdir(category_path)):
                        if file.endswith(".json") and not file.startswith("._"):
                            full_path = os.path.join(category_path, file)
                            result[subdir][category].append(full_path)
    return result

def collect_atlas_label_paths(base_dir):
    result = dict()
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".txt"):
                    full_path = os.path.join(subdir_path, file)
                    name_only = os.path.splitext(file)[0]
                    result[name_only] = full_path
    return result

def collect_label_paths(base_dir):
    """Return every raw-data label file per scene in deterministic order."""
    result = defaultdict(list)
    for subdir in sorted(os.listdir(base_dir)):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            category_path = os.path.join(subdir_path, "malicious")
            if os.path.exists(category_path):
                for file in sorted(os.listdir(category_path)):
                    if file.endswith(".txt") and not file.startswith("._"):
                        full_path = os.path.join(category_path, file)
                        result[subdir].append(full_path)
    return dict(result)


def _read_uuid_file(path: Path) -> set[str]:
    values: set[str] = set()
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                for key in ("actorID", "objectID", "uuid", "name"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        values.add(value)
    else:
        with path.open("r", encoding="utf-8") as stream:
            values.update(line.strip() for line in stream if line.strip())
    return values


def load_released_malicious_uuids(
    dataset_name: str,
    scene_name: Optional[str] = None,
) -> set[str]:
    """Load repository entity labels within the selected dataset scope.

    Paper-profile OpTC execution is restricted to H051/H201/H501. DARPA E5
    loads the explicit PIDSMaker attack sources for one platform; these UUIDs
    are node labels and are not interpreted as event IDs or ATT&CK boundaries.
    """
    dataset_name = str(dataset_name or "").lower()
    label_dir = _RELEASED_LABEL_DIR.get(dataset_name)
    if label_dir is None:
        return set()
    root = REPO_ROOT / "data" / "annotated_labels" / label_dir / "malicious_entities"
    if dataset_name == "optcday1":
        paths = [root / name for name in _OPTC_PAPER_LABEL_FILES]
    elif dataset_name in _E5_PIDSMaker_LABEL_FILES:
        paths = [root / name for name in _E5_PIDSMaker_LABEL_FILES[dataset_name]]
    elif scene_name:
        paths = [root / f"{scene_name}.txt", root / f"{scene_name}.csv"]
    else:
        prefix = _RELEASED_LABEL_PREFIX[dataset_name]
        paths = sorted(root.glob(f"{prefix}*.txt")) + sorted(root.glob(f"{prefix}*.csv"))
    labels: set[str] = set()
    for path in paths:
        if path.exists():
            labels.update(_read_uuid_file(path))
    return labels


def load_optc_released_malicious_uuids_by_host() -> dict[str, set[str]]:
    """Return paper-profile OpTC node labels in their original host scope.

    OpTC node UUIDs and audit-event UUIDs are different namespaces.  The
    released files contain only PIDSMaker actor/object node UUIDs, and the
    host key is retained so an identifier observed on one machine cannot
    label a node on another machine.
    """
    root = REPO_ROOT / "data" / "annotated_labels" / "optc" / "malicious_entities"
    result = {}
    for host_id, filename in _OPTC_PAPER_LABEL_BY_HOST.items():
        path = root / filename
        result[host_id] = _read_uuid_file(path) if path.exists() else set()
    return result


def snapshot_local_property(row, action: str, node_role: str, node_type) -> str:
    """Build a node property only from events present in the current window.

    The old parsers looked up every node in a corpus-wide property dictionary.
    That allowed an attribute observed in a later held-out event to appear in an
    earlier training snapshot.  This helper intentionally uses only fields
    attached to the current event rows. Missing fields receive a typed marker,
    rather than being back-filled from future corpus records.
    """
    role = str(node_role).lower()
    entity_type = str(node_type or "").lower()
    command = str(getattr(row, "exec", "") or "").strip()
    generic_path = str(getattr(row, "path", "") or "").strip()
    actor_path = str(getattr(row, "actor_path", "") or "").strip()
    object_path = str(getattr(row, "object_path", "") or "").strip()
    path = actor_path if role == "actor" else (object_path or generic_path)

    if "process" in entity_type or "subject" in entity_type:
        local_command = command if role == "actor" and command else "[unknown-process]"
        return f"{local_command},,{path or '[unknown-path]'}"
    if "file" in entity_type:
        return path or f"[unknown-file]:{action}"
    if any(token in entity_type for token in ("net", "flow", "sock")):
        return path or f"[snapshot-local-network]:{action}"
    return path or f"[snapshot-local-{entity_type or 'entity'}]:{action}"


def _unwrap_avro_scalar(value) -> str:
    if isinstance(value, dict):
        for key in (
            "com.bbn.tc.schema.avro.cdm18.UUID",
            "com.bbn.tc.schema.avro.cdm20.UUID",
            "string",
            "uuid",
        ):
            if key in value:
                return _unwrap_avro_scalar(value[key])
        if len(value) == 1:
            return _unwrap_avro_scalar(next(iter(value.values())))
        return ""
    return str(value or "").strip()


_CDM_UUID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-?[0-9A-Fa-f]{4}-?[0-9A-Fa-f]{4}-?"
    r"[0-9A-Fa-f]{4}-?[0-9A-Fa-f]{12}$"
)


def normalize_cdm_uuid(value) -> str:
    """Normalize CDM UUID case/hyphen variants without changing other IDs."""
    text = _unwrap_avro_scalar(value).strip().strip("{}")
    if not _CDM_UUID_RE.fullmatch(text):
        return text
    compact = text.replace("-", "").upper()
    return "-".join((compact[:8], compact[8:12], compact[12:16], compact[16:20], compact[20:]))


def cdm_host_identity(record: dict, event: dict, source_path) -> tuple[str, str]:
    """Extract a stable per-event host identity without a global default.

    CDM releases differ in whether ``hostId`` is attached to the event, datum,
    or envelope.  If an export omits it entirely, the exact source shard path
    is retained as the namespace instead of silently merging hosts.
    """
    datum = record.get("datum") if isinstance(record, dict) else None
    candidates = [
        event.get("hostId") if isinstance(event, dict) else None,
        datum.get("hostId") if isinstance(datum, dict) else None,
        record.get("hostId") if isinstance(record, dict) else None,
    ]
    for candidate in candidates:
        value = normalize_cdm_uuid(candidate)
        if value:
            return value, "cdm.hostId"
    path = Path(source_path)
    parent = path.parent
    if parent.name.lower() in {"benign", "malicious", "logs", "events"}:
        parent = parent.parent
    scene_id = parent.name.strip()
    if not scene_id:
        raise RuntimeError(f"cannot derive stable host scene directory for {source_path}")
    # A scene directory is stable across machines and merges multiple source
    # shards belonging to the same single-host DARPA scene.
    return f"scene-dir:{scene_id}", "scene_directory"


def cdm_host_id(record: dict, event: dict, source_path) -> str:
    return cdm_host_identity(record, event, source_path)[0]


def cdm_event_id(event: dict, source_path, record_number: int, endpoint: str) -> str:
    for key in ("uuid", "eventId", "id"):
        value = normalize_cdm_uuid(event.get(key)) if isinstance(event, dict) else ""
        if value:
            return f"{value}:{endpoint}"
    path = Path(source_path)
    scene = path.parent.parent.name if path.parent.name.lower() in {"benign", "malicious"} else path.parent.name
    payload = "\x1f".join([
        scene,
        path.name,
        str(int(record_number)),
        str(event.get("timestampNanos", "")),
        endpoint,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_event_id(row, ordinal: int, namespace: str) -> str:
    supplied = str(getattr(row, "event_id", "") or "").strip()
    if supplied:
        return supplied
    values = [
        namespace,
        str(getattr(row, "host_id", "") or ""),
        str(getattr(row, "actorID", "") or ""),
        str(getattr(row, "objectID", "") or ""),
        str(getattr(row, "action", "") or ""),
        str(getattr(row, "timestamp", "") or ""),
        str(int(ordinal)),
    ]
    return hashlib.sha256("\x1f".join(values).encode("utf-8")).hexdigest()


def add_typed_event_edges(
    graph, index_map: dict, rows, namespace: str, timestamp_scale: float = 1.0,
):
    """Add one directed typed edge per event row, including parallel edges."""
    events = []
    for ordinal, row in enumerate(rows):
        actor = getattr(row, "actorID")
        obj = getattr(row, "objectID")
        raw_ts = getattr(row, "timestamp")
        timestamp_dt = getattr(row, "timestamp_dt", None)
        if timestamp_dt is not None and pd.notna(timestamp_dt):
            timestamp = timestamp_dt.timestamp()
        else:
            try:
                timestamp = float(raw_ts) / float(timestamp_scale)
            except (TypeError, ValueError):
                timestamp = pd.to_datetime(raw_ts, utc=True).timestamp()
        events.append({
            "source": index_map[actor],
            "target": index_map[obj],
            "actions": str(getattr(row, "action")),
            "timestamp": timestamp,
            "event_id": stable_event_id(row, ordinal, namespace),
            "input_order": ordinal,
            "source_event_id": str(getattr(row, "source_event_id", "") or ""),
            "replay_source_snapshot": getattr(row, "replay_source_snapshot", None),
            "replay_attack_id": str(getattr(row, "replay_attack_id", "") or ""),
            "replay_condition": str(getattr(row, "replay_condition", "") or ""),
        })
    # Source order is the causal tie-breaker for simultaneous events.  Event
    # IDs are identifiers, not temporal values, and must never reorder them.
    events.sort(key=lambda item: (item["timestamp"], item["input_order"], item["event_id"]))
    if events:
        graph.add_edges([(item["source"], item["target"]) for item in events])
        graph.es["actions"] = [item["actions"] for item in events]
        graph.es["timestamp"] = [item["timestamp"] for item in events]
        graph.es["event_id"] = [item["event_id"] for item in events]
        graph.es["event_order"] = list(range(len(events)))
        if any(item["source_event_id"] for item in events):
            graph.es["source_event_id"] = [item["source_event_id"] for item in events]
            graph.es["replay_source_snapshot"] = [item["replay_source_snapshot"] for item in events]
            graph.es["replay_attack_id"] = [item["replay_attack_id"] for item in events]
            graph.es["replay_condition"] = [item["replay_condition"] for item in events]
    edge_index = [
        [item["source"] for item in events],
        [item["target"] for item in events],
    ]
    relations_index = {
        (item["source"], item["target"], order): item["actions"]
        for order, item in enumerate(events)
    }
    return edge_index, relations_index


def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] =set()
    nodes[node_id].add(properties)


def get_or_add_node(G, node_id, node_type, properties):
    """
    findgraphinisalreadyhasnode node_id: 
    - ifhas, returnnodeindex, updateattribute
    - ifno, addnodereturnitsindex
    """
    try:
        v = G. vs .find(name=node_id)
        v['properties'] = properties
        return v.index
    except ValueError:
        G.add_vertex(name=node_id, type=node_type, properties=properties)
        return len(G. vs ) - 1

def add_edge_if_new(G, src, dst, action):
    """
    towardgraph G add1entryfrom src to dst 's edge,  action attribute. 
    - ifedgealreadyexistscontains action, notanyprocess. 
    - ifedgealreadyexistsbutcontains action againadd1entryedge
    - ifedgedoes not exist, thenaddedgeset action. 
    """
    if G.are_connected(src, dst):
        eids = G.get_eids([(src, dst)], directed=True, error=False)
        for eid in eids:
            if G.es[eid]["actions"] == action:
                return  #  action alreadyexists, notadd
    G.add_edge(src, dst)
    G.es[-1]["actions"] = action

def update_edge_index(edges, edge_index, index, relations, relations_index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)

        relation = relations[(src_id, dst_id)]
        relations_index[(src, dst)] = relation
