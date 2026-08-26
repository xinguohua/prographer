"""ATLAS indexed-event loader with the original ten-fold scenario protocol.

ATLAS defines four single-host scenarios (S1--S4) and six multi-host
scenarios (M1--M6).  A requested fold loads every scenario in the same family;
``src.utils.split`` then holds the requested scenario out and trains on the
remaining three or five scenarios respectively.
"""
from __future__ import annotations

import re
import hashlib
import json
from pathlib import Path
from typing import Optional

import igraph as ig
import numpy as np
import pandas as pd

from ._base import BaseProcessor
from .atlas_v1 import (
    convert_official_case,
    convert_preprocessed_file,
    discover_official_cases,
    discover_preprocessed_files,
    host_from_name,
    load_case_labels,
    normalize_atlas_value,
    resolve_case_label_ids,
)


ATLAS_SINGLE_FOLDS = ("S1", "S2", "S3", "S4")
ATLAS_MULTI_FOLDS = ("M1", "M2", "M3", "M4", "M5", "M6")
ATLAS_FOLDS = ATLAS_SINGLE_FOLDS + ATLAS_MULTI_FOLDS

_SCENARIO_RE = re.compile(r"(?<![A-Z0-9])([SM][1-6])(?![0-9])", re.IGNORECASE)
_HOST_RE = re.compile(r"(?<![A-Z0-9])(H\d+|EDR)(?![A-Z0-9])", re.IGNORECASE)


def atlas_scenario_from_path(path: Path) -> Optional[str]:
    """Return the S/M scenario identifier encoded in a path."""
    for value in (path.stem, *reversed(path.parts)):
        match = _SCENARIO_RE.search(str(value).replace("_", "-"))
        if match:
            scenario = match.group(1).upper()
            if scenario in ATLAS_FOLDS:
                return scenario
    return None


def atlas_host_from_path(path: Path) -> str:
    for value in (path.stem, *reversed(path.parts)):
        match = _HOST_RE.search(str(value).replace("_", "-"))
        if match:
            return match.group(1).upper()
    return "H1"


def _read_event_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    frame = pd.read_csv(path, sep=sep, low_memory=False)
    aliases = {
        "actorid": "actorID",
        "actor_type": "actor_type",
        "objectid": "objectID",
        "object": "object",
        "object_type": "object",
        "action": "action",
        "timestamp": "timestamp",
        "source_id": "actorID",
        "source_type": "actor_type",
        "destination_id": "objectID",
        "destination_type": "object",
        "edge_type": "action",
        "time": "timestamp",
        "time_unit": "timestamp_unit",
        "timestampunit": "timestamp_unit",
        "eventid": "event_id",
        "command_line": "command",
        "args": "arguments",
        "file_path": "path",
        "network_address": "address",
    }
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    renames = {}
    for alias, target in aliases.items():
        original = normalized.get(alias)
        if original is not None and target not in frame.columns:
            renames[original] = target
    frame = frame.rename(columns=renames)
    required = (
        "actorID", "actor_type", "objectID", "object", "action", "timestamp", "timestamp_unit",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"ATLAS event table {path} is missing columns: {missing}")
    optional = (
        "event_id", "command", "arguments", "path", "address",
        "src_address", "src_port", "dst_address", "dst_port",
    )
    for column in optional:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame.loc[:, list(required) + list(optional)].copy()
    for column in required[:-2] + optional:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["timestamp_unit"] = frame["timestamp_unit"].fillna("").astype(str).str.strip().str.lower()
    valid_units = {"s", "ms", "us", "ns"}
    invalid_units = sorted(set(frame["timestamp_unit"]) - valid_units)
    if invalid_units:
        raise ValueError(f"ATLAS event table {path} has invalid timestamp_unit: {invalid_units}")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame[(frame["actorID"] != "") & (frame["objectID"] != "")]
    for index in range(len(frame)):
        if not frame.iloc[index]["event_id"]:
            payload = "\x1f".join(
                [path.name, str(index), *map(str, frame.iloc[index][list(required)].tolist())]
            )
            frame.iat[index, frame.columns.get_loc("event_id")] = hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()
    if frame["event_id"].duplicated().any():
        raise ValueError(f"ATLAS event table {path} contains duplicate event_id values")
    return frame.reset_index(drop=True)


def _timestamp_seconds(values: pd.Series, units: pd.Series) -> pd.Series:
    """Convert explicitly declared ATLAS timestamp units to seconds."""
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    scale = units.astype(str).str.lower().map({"s": 1.0, "ms": 1e3, "us": 1e6, "ns": 1e9})
    if scale.isna().any():
        raise ValueError("ATLAS timestamps require an explicit s/ms/us/ns unit")
    return numeric / scale


def _local_label_paths(root: Path, scenario: str) -> list[Path]:
    scenario_root = root / scenario
    paths = [scenario_root / "malicious_labels.txt"]
    return [path for path in paths if path.is_file()]


def _load_local_labels(root: Path, scenario: str) -> set[str]:
    labels = set()
    for path in _local_label_paths(root, scenario):
        with path.open("r", encoding="utf-8", errors="ignore") as stream:
            labels.update(line.strip().lower() for line in stream if line.strip())
    return labels


def _preprocessed_case_labels(root: Path, path: Path) -> tuple[Path, set[str]]:
    name = path.name
    case_name = name
    for prefix in ("training_preprocessed_logs_", "testing_preprocessed_logs_"):
        if name.startswith(prefix):
            case_name = name[len(prefix):]
            break
    candidates = []
    experiment_roots = [root, root / "paper_experiments"]
    for ancestor in path.parents:
        try:
            ancestor.relative_to(root)
        except ValueError:
            continue
        experiment_roots.append(ancestor)
    for experiment_root in dict.fromkeys(experiment_roots):
        for split_name in ("training_logs", "testing_logs"):
            candidates.append(experiment_root / split_name / case_name)
    for case_dir in candidates:
        label_path = case_dir / "malicious_labels.txt"
        if label_path.is_file():
            return case_dir, load_case_labels(case_dir)
    raise RuntimeError(
        f"ATLAS preprocessed file {path} requires its same-case malicious_labels.txt"
    )


class ATLASHandler(BaseProcessor):
    """Build one-minute ATHENA snapshots for one ATLAS leave-one-out fold."""

    def __init__(
        self,
        base_path,
        train,
        *,
        scene_name: Optional[str] = None,
        dataset_name: str = "atlas",
        source_timezone: str = "-05:00",
    ):
        super().__init__(base_path, train)
        fold = str(scene_name or "").upper()
        if fold not in ATLAS_FOLDS:
            raise ValueError(
                "ATLAS requires --scene set to one original fold: " + ", ".join(ATLAS_FOLDS)
            )
        self.dataset_name = dataset_name
        self.source_timezone = str(source_timezone)
        self.scene_name = fold
        self.atlas_fold = fold
        self.atlas_family = fold[0]
        self.atlas_scenarios = ATLAS_SINGLE_FOLDS if fold.startswith("S") else ATLAS_MULTI_FOLDS
        self.event_tables: dict[str, pd.DataFrame] = {}
        self.scenario_labels: dict[str, set[str]] = {}
        self.snapshot_scenarios: list[str] = []
        self.snapshot_hosts: list[str] = []
        self.label_coverage_audit: list[dict] = []

    def load(self):
        root = Path(self.base_path)
        if not root.exists():
            raise FileNotFoundError(f"ATLAS dataset root does not exist: {root}")
        grouped: dict[str, list[pd.DataFrame]] = {scenario: [] for scenario in self.atlas_scenarios}
        labels_by_scenario: dict[str, set[str]] = {scenario: set() for scenario in self.atlas_scenarios}
        preprocessed_files = discover_preprocessed_files(root, self.atlas_scenarios)
        official_cases = discover_official_cases(root, self.atlas_scenarios)
        if preprocessed_files:
            for path in preprocessed_files:
                scenario = atlas_scenario_from_path(path)
                if scenario not in grouped:
                    continue
                frame = convert_preprocessed_file(path)
                frame["_atlas_scenario"] = scenario
                frame["_atlas_host"] = atlas_host_from_path(path)
                case_dir, source_labels = _preprocessed_case_labels(root, path)
                resolved, audit = resolve_case_label_ids(case_dir, source_labels, frame)
                endpoint_positive = frame["matched_labels"].astype(str).ne("[]")
                suffix_positive = frame["malicious_event"].astype(bool)
                labels_by_scenario[scenario].update(resolved)
                audit.update({
                    "scenario": scenario,
                    "host": atlas_host_from_path(path),
                    "case": path.name,
                    "label_source": str(path),
                    "label_mode": "official_normalized_endpoint_substring",
                    "suffix_positive_events": int(suffix_positive.sum()),
                    "endpoint_positive_events": int(endpoint_positive.sum()),
                    "endpoint_only_positive_events": int(
                        (endpoint_positive & ~suffix_positive).sum()
                    ),
                    "suffix_only_positive_events": int(
                        (suffix_positive & ~endpoint_positive).sum()
                    ),
                })
                self.label_coverage_audit.append(audit)
                grouped[scenario].append(frame)
        elif official_cases:
            for case_dir in official_cases:
                scenario = atlas_scenario_from_path(case_dir)
                if scenario not in grouped:
                    continue
                frame = convert_official_case(case_dir, self.source_timezone)
                frame["_atlas_scenario"] = scenario
                frame["_atlas_host"] = host_from_name(case_dir)
                source_labels = load_case_labels(case_dir)
                resolved, audit = resolve_case_label_ids(case_dir, source_labels, frame)
                audit.update({"scenario": scenario, "host": host_from_name(case_dir)})
                self.label_coverage_audit.append(audit)
                labels_by_scenario[scenario].update(resolved)
                grouped[scenario].append(frame)
        else:
            for scenario in self.atlas_scenarios:
                scenario_root = root / scenario
                if not scenario_root.is_dir():
                    continue
                paths = sorted(scenario_root.glob("*_events.csv")) + sorted(
                    scenario_root.glob("*_events.tsv")
                )
                for path in paths:
                    frame = _read_event_table(path)
                    if frame.empty:
                        continue
                    frame["_atlas_scenario"] = scenario
                    frame["_atlas_host"] = atlas_host_from_path(path)
                    grouped[scenario].append(frame)
                labels_by_scenario[scenario].update(_load_local_labels(root, scenario))

        missing = [scenario for scenario, parts in grouped.items() if not parts]
        if missing:
            raise RuntimeError(
                f"ATLAS fold {self.atlas_fold} requires indexed event tables for "
                f"{', '.join(self.atlas_scenarios)}; missing {', '.join(missing)}"
            )

        self.all_labels = []
        for scenario in self.atlas_scenarios:
            frame = pd.concat(grouped[scenario], ignore_index=True)
            if frame["event_id"].duplicated().any():
                raise RuntimeError(f"ATLAS scenario {scenario} contains duplicate event IDs")
            self.event_tables[scenario] = frame
            entities = {
                normalize_atlas_value(value)
                for value in set(frame["actorID"]) | set(frame["objectID"])
            }
            if preprocessed_files or official_cases:
                labels = labels_by_scenario[scenario] & entities
            else:
                source_labels = {
                    normalize_atlas_value(value) for value in labels_by_scenario[scenario]
                }
                labels = {
                    entity for entity in entities
                    if any(label and label in entity for label in source_labels)
                }
                frame["matched_labels"] = [
                    json.dumps(sorted({
                        label
                        for label in source_labels
                        if label and (
                            label in normalize_atlas_value(row.actorID)
                            or label in normalize_atlas_value(row.objectID)
                        )
                    }))
                    for row in frame.itertuples(index=False)
                ]
                frame["label_source"] = "indexed_scenario_malicious_labels.txt"
                frame["malicious_event"] = frame["matched_labels"].ne("[]")
            if not labels:
                raise RuntimeError(
                    f"ATLAS scenario {scenario} has no exact malicious-label/entity intersection"
                )
            self.scenario_labels[scenario] = labels
            self.all_labels.extend(sorted(labels))
        self.all_labels = sorted(set(self.all_labels))
        self.malicious = pd.concat(
            [self.event_tables[scenario] for scenario in self.atlas_scenarios],
            ignore_index=True,
        )
        self.begin = self.malicious.iloc[0:0].copy()

    @staticmethod
    def _matches_label(node_id: str, labels: set[str]) -> bool:
        return normalize_atlas_value(node_id) in labels

    def create_snapshots_from_graph(self, df, is_malicious):
        if df is None or len(df) == 0:
            return []
        snapshots = []
        for (scenario, host), group in df.groupby(
            ["_atlas_scenario", "_atlas_host"], sort=True,
        ):
            group = group.copy()
            group["timestamp_seconds"] = _timestamp_seconds(
                group["timestamp"], group["timestamp_unit"],
            )
            group = group.dropna(subset=["timestamp_seconds"])
            group["window_id"] = np.floor(group["timestamp_seconds"] / 60.0).astype(np.int64)
            labels = self.scenario_labels.get(str(scenario), set())
            for window, chunk in group.groupby("window_id", sort=True):
                chunk = chunk.sort_values(["timestamp_seconds", "event_id"], kind="stable").copy()
                chunk["event_order"] = np.arange(len(chunk), dtype=np.int64)
                graph = ig.Graph(directed=True)
                node_rows: dict[str, list] = {}
                for row in chunk.itertuples(index=False):
                    node_rows.setdefault(str(row.actorID), []).append((row, "actor", row.actor_type))
                    node_rows.setdefault(str(row.objectID), []).append((row, "object", row.object))
                for node_id, rows in node_rows.items():
                    row, role, entity_type = rows[0]
                    node_matched_labels = set()
                    label_sources = set()
                    for semantic_row, _semantic_role, _semantic_type in rows:
                        try:
                            matched = json.loads(str(getattr(semantic_row, "matched_labels", "[]") or "[]"))
                        except json.JSONDecodeError:
                            matched = []
                        node_matched_labels.update(
                            label for label in matched
                            if normalize_atlas_value(label) in normalize_atlas_value(node_id)
                        )
                        source = str(getattr(semantic_row, "label_source", "") or "")
                        if source:
                            label_sources.add(source)
                    semantic_events = []
                    for semantic_row, semantic_role, _semantic_type in rows:
                        semantic_events.append({
                            "event_id": str(semantic_row.event_id),
                            "event_order": int(getattr(semantic_row, "event_order", -1)),
                            "role": semantic_role,
                            "action": str(semantic_row.action),
                            "command": str(getattr(semantic_row, "command", "") or ""),
                            "arguments": str(getattr(semantic_row, "arguments", "") or ""),
                            "path": str(getattr(semantic_row, "path", "") or ""),
                            "address": str(getattr(semantic_row, "address", "") or ""),
                        })
                    graph.add_vertex(
                        name=node_id,
                        _athena_temporal_id=f"atlas:{scenario}:{host}:{node_id}",
                        type=str(entity_type),
                        properties=json.dumps({
                            "entity_type": str(entity_type),
                            "events": semantic_events,
                        }, ensure_ascii=False, sort_keys=True),
                        label=int(self._matches_label(node_id, labels)),
                        matched_labels=sorted(node_matched_labels),
                        label_source=sorted(label_sources),
                        frequency=len(rows),
                        timestamp=float(chunk["timestamp_seconds"].min()),
                    )
                index = {str(vertex["name"]): vertex.index for vertex in graph.vs}
                for row in chunk.itertuples(index=False):
                    graph.add_edge(
                        index[str(row.actorID)],
                        index[str(row.objectID)],
                        actions=str(row.action),
                        timestamp=float(row.timestamp_seconds),
                        event_id=str(row.event_id),
                        event_order=int(row.event_order),
                        command=str(getattr(row, "command", "") or ""),
                        arguments=str(getattr(row, "arguments", "") or ""),
                        path=str(getattr(row, "path", "") or ""),
                        address=str(getattr(row, "address", "") or ""),
                        malicious_event=bool(getattr(row, "malicious_event", False)),
                        matched_labels=str(getattr(row, "matched_labels", "[]") or "[]"),
                        label_source=str(getattr(row, "label_source", "") or ""),
                    )
                graph["atlas_scenario"] = str(scenario)
                graph["source_scene"] = str(scenario)
                graph["atlas_host"] = str(host)
                graph["host_id"] = str(host)
                graph["host_id_source"] = "atlas_case_or_indexed_filename"
                graph["window_start"] = float(window) * 60.0
                snapshots.append(graph)
        return snapshots

    def build_graph(self, gid=None):
        self.snapshots = []
        self.snapshot_scenarios = []
        self.snapshot_hosts = []
        for scenario in self.atlas_scenarios:
            for graph in self.create_snapshots_from_graph(self.event_tables[scenario], True):
                self.snapshots.append(graph)
                self.snapshot_scenarios.append(str(graph["atlas_scenario"]))
                self.snapshot_hosts.append(str(graph["atlas_host"]))
        self.benign_idx_start = 0 if self.snapshots else -1
        self.benign_idx_end = len(self.snapshots) - 1
        self.malicious_idx_start = 0 if self.snapshots else -1
        self.malicious_idx_end = len(self.snapshots) - 1


__all__ = [
    "ATLASHandler",
    "ATLAS_FOLDS",
    "ATLAS_SINGLE_FOLDS",
    "ATLAS_MULTI_FOLDS",
    "atlas_scenario_from_path",
    "atlas_host_from_path",
]
