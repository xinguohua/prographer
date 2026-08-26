"""Global attack interpretation: technique mapping + tactic-level alignment.

For each snapshot marked malicious in the loaded dataset:
1. Extract the key causal sub-path rooted at every malicious node.
2. Translate the sub-path into a natural-language query
   (:mod:`src.interpretation.semantic_matching`).
3. Retrieve a top-K parent-level ATT&CK technique/tactic set per attack
   subgraph with Sentence-BERT cosine similarity.
4. Append each candidate set to the host's persistent queue, beam-compose
   top-K tactic chains, and LCS/min-align every chain against AttackSeqBench.

Usage:
    python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection.node_labels import load_malicious_uuids
from src.interpretation.attack_subgraph import reconstruct_attack_paths
from src.interpretation.global_alignment import align_candidate_chains, build_candidate_tactic_chains
from src.interpretation.semantic_matching import TechniqueSemanticMapper, path_edges_to_query
from src.interpretation.tactic_alignment import (
    load_tactic_sequence_records,
    load_tech_to_tactics,
    normalize_tech_id,
)
from src.snapshot_construction.graph_loader import get_handler
from src.snapshot_construction._common import normalize_cdm_uuid
from src.utils.config import load_config
from src.utils.interval_replay import apply_interval_replay, event_sha256
from src.utils.split import ATLAS_SPLIT_MODE, SPLIT_MODE, build_split


SUPPORTED_DATASETS = (
    "cadets", "theia", "trace", "clearscope",
    "cadets5", "theia5", "trace5", "clearscope5",
    "optcday1",
    "atlas",
)
MAPPING_VARIANTS = ("direct", "tech-enhanced", "log-enhanced", "full-enhanced")
E5_SOURCE_DATASET = {
    "cadets5": "cadets",
    "theia5": "theia",
    "trace5": "trace",
    "clearscope5": "clearscope",
}


def _mapping_variant_settings(variant: str) -> tuple[str, bool]:
    if variant not in MAPPING_VARIANTS:
        raise ValueError(f"unknown mapping variant: {variant}")
    technique_file = (
        "technique_triples_transformed.json"
        if variant in {"tech-enhanced", "full-enhanced"}
        else "technique_triples_raw.json"
    )
    log_enhanced = variant in {"log-enhanced", "full-enhanced"}
    return technique_file, log_enhanced


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "athena.yaml"))
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p.add_argument(
        "--scene", default=None,
        help="scene filter; for ATLAS, the held-out original fold S1-S4 or M1-M6",
    )
    p.add_argument("--top-k", type=int, default=None,
                   help="override interpretation.topk_candidates from the config")
    p.add_argument(
        "--mapping-variant", choices=MAPPING_VARIANTS, default="full-enhanced",
        help="Table VII technique/log enhancement branch",
    )
    p.add_argument("--max-malicious", type=int, default=None,
                   help="cap the number of malicious snapshots interpreted (smoke testing)")
    p.add_argument("--output", default=None,
                   help="optional JSON file to dump per-snapshot interpretation + alignment")
    p.add_argument("--detections", default=None,
                   help="JSON produced by scripts/run_detection.py; used as the interpretation input")
    p.add_argument("--benign-injection-manifest", type=Path,
                   help="same source-linked E3 interval replay manifest used by detection")
    p.add_argument("--attack-event-boundaries", type=Path,
                   help="source-linked Table VIII attack-case boundary JSONL")
    p.add_argument("--include-train-detections", action="store_true",
                   help="also interpret detector positives from training snapshots; default uses held-out test positives only")
    p.add_argument("--use-ground-truth", action="store_true",
                   help="debug/evaluation mode: interpret released malicious UUIDs instead of detector output")
    return p.parse_args(argv)


def _malicious_nodes(snap, malicious_uuids: set) -> list:
    """Return igraph vertex indices that are malicious.

    A node is malicious if either its ``label`` attribute equals 1 (set by
    the augmented training pipeline on ego subgraphs) or its ``name`` (CDM
    UUID) is in the released malicious-entity set for the scene.
    """
    mal = []
    for v in range(snap.vcount()):
        attrs = snap.vs[v].attributes()
        try:
            if int(attrs.get("label", 0)) == 1:
                mal.append(v)
                continue
        except (TypeError, ValueError):
            pass
        name = str(attrs.get("name", ""))
        if name and name in malicious_uuids:
            mal.append(v)
    return mal


def _mark_malicious(snap, malicious_uuids: set) -> int:
    """Set ``label=1`` on every snapshot vertex whose ``name`` UUID is in the
    released malicious-entity set so downstream helpers that key off
    ``label`` (e.g. :func:`snapshot_to_query`) see them. Returns the count
    of vertices newly labelled."""
    if not malicious_uuids:
        return 0
    count = 0
    for v in range(snap.vcount()):
        attrs = snap.vs[v].attributes()
        name = str(attrs.get("name", ""))
        if name in malicious_uuids:
            try:
                if int(attrs.get("label", 0)) != 1:
                    snap.vs[v]["label"] = 1
                    count += 1
            except (TypeError, ValueError):
                snap.vs[v]["label"] = 1
                count += 1
    return count


def _load_detected_nodes(
    path: str,
    include_train: bool = False,
    *,
    expected_dataset: str,
    expected_scene,
) -> dict:
    """Load detector positives from ``scripts/run_detection.py`` output.

    Returns ``{snapshot_id: set(uuid)}``. Ground-truth labels in the detection
    file are ignored here; they are only for metric reporting.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("dataset") != expected_dataset or payload.get("scene") != expected_scene:
        raise RuntimeError("detection JSON dataset/scene does not match interpretation request")
    split = payload.get("split")
    if not isinstance(split, dict) or not split.get("test_snapshots"):
        raise RuntimeError("detection JSON lacks an auditable held-out split contract")
    split_mode = split.get("mode")
    if split_mode not in {SPLIT_MODE, ATLAS_SPLIT_MODE, "e3-checkpoint-transfer-eval"}:
        raise RuntimeError("detection JSON uses an unsupported split mode")
    if split_mode == "e3-checkpoint-transfer-eval":
        checkpoint = payload.get("checkpoint") if isinstance(payload.get("checkpoint"), dict) else {}
        expected_source = E5_SOURCE_DATASET.get(expected_dataset)
        if (
            payload.get("execution") != "eval-only"
            or checkpoint.get("source_dataset") not in {"cadets", "theia", "trace", "clearscope"}
            or (expected_source is not None and checkpoint.get("source_dataset") != expected_source)
            or checkpoint.get("source_run_mode") != "complete"
            or checkpoint.get("source_variant") != "full-athena"
            or not (checkpoint.get("source_augmentation") or {}).get("manifest_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(checkpoint.get("sha256", "")))
            or split.get("source_training") != checkpoint
            or split.get("train_snapshots")
        ):
            raise RuntimeError("transfer-eval detection lacks bound E3 checkpoint provenance")
    train_ids = {int(value) for value in split.get("train_snapshots", [])}
    test_ids = {int(value) for value in split.get("test_snapshots", [])}
    if train_ids & test_ids:
        raise RuntimeError("detection JSON split contract overlaps train and test snapshots")
    out = {}
    for row in payload.get("predictions", []):
        if split_mode == "e3-checkpoint-transfer-eval" and row.get("split") != "test":
            raise RuntimeError("transfer-eval predictions must contain held-out test rows only")
        if int(row.get("pred_label", 0)) != 1:
            continue
        row_split = row.get("split")
        if row_split not in {"train", "test"}:
            raise RuntimeError("detection prediction lacks an explicit train/test split label")
        if not include_train and row_split != "test":
            continue
        sid = int(row["snapshot"])
        expected_ids = test_ids if row_split == "test" else train_ids
        if sid not in expected_ids:
            raise RuntimeError("prediction split label contradicts the detection split contract")
        out.setdefault(sid, set()).add(str(row["uuid"]))
    return out


def _mark_detected(snap, detected_uuids: set) -> int:
    count = 0
    for v in range(snap.vcount()):
        attrs = snap.vs[v].attributes()
        name = str(attrs.get("name", ""))
        is_detected = name in detected_uuids
        snap.vs[v]["label"] = 1 if is_detected else 0
        if is_detected:
            count += 1
    return count


def _snapshot_time_seconds(snap, fallback_index: int) -> float:
    """Return the latest timestamp observed in a snapshot.

    DARPA CDM timestamps are nanoseconds in the released parsers; OpTC stores
    seconds. If a snapshot has no usable timestamp, fall back to snapshot order
    at one-minute spacing, matching the paper's window construction.
    """
    values = []
    for seq in (getattr(snap, "vs", []), getattr(snap, "es", [])):
        try:
            for item in seq:
                attrs = item.attributes()
                if "timestamp" in attrs:
                    values.append(float(attrs.get("timestamp") or 0.0))
        except (TypeError, ValueError):
            continue
    if not values:
        return float(fallback_index) * 60.0
    ts = max(values)
    return ts / 1_000_000_000.0 if ts > 1e12 else ts


def _snapshot_host_id(snap) -> str:
    attrs = set(snap.attributes()) if hasattr(snap, "attributes") else set()
    if "host_id" not in attrs:
        raise RuntimeError("snapshot is missing the stable host_id required by the persistent queue")
    host_id = str(snap["host_id"] or "").strip()
    if not host_id:
        raise RuntimeError("snapshot host_id is empty")
    return host_id


def _snapshot_source_scene(snap, requested_scene: str | None) -> str:
    attrs = set(snap.attributes()) if hasattr(snap, "attributes") else set()
    source_scene = str(snap["source_scene"] or "").strip() if "source_scene" in attrs else ""
    if not source_scene:
        source_scene = str(requested_scene or "").strip()
    if not source_scene:
        raise RuntimeError("snapshot is missing source_scene required for RQ3 ground-truth joins")
    return source_scene


def _node_causal_order(snap, node_index: int):
    keys = []
    for edge_id in snap.incident(int(node_index), mode="ALL"):
        attrs = snap.es[int(edge_id)].attributes()
        try:
            timestamp = float(attrs.get("timestamp", 0.0) or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        try:
            raw_order = attrs.get("event_order", edge_id)
            event_order = int(edge_id if raw_order in (None, "") else raw_order)
        except (TypeError, ValueError):
            event_order = int(edge_id)
        keys.append((timestamp, event_order, int(edge_id)))
    if keys:
        return min(keys) + (int(node_index),)
    attrs = snap.vs[int(node_index)].attributes()
    try:
        timestamp = float(attrs.get("timestamp", 0.0) or 0.0)
    except (TypeError, ValueError):
        timestamp = 0.0
    return timestamp, 0, -1, int(node_index)


def _prediction_commit_order(snap, node_index: int, position=None):
    """Use the exact registry event for scored predictions, not the node's earliest edge."""
    if position is not None:
        timestamp, event_order, snapshot_id, _edge_index = position
        return float(timestamp), int(event_order), int(snapshot_id), int(node_index)
    return _node_causal_order(snap, node_index)


def _canonical_record_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _portable_boundary_evidence(source_record: Path, row: dict, boundary: dict) -> None:
    """Bind a portable event/anchor boundary to its exact public source row."""
    record_hash = str(row.get("source_record_sha256") or "").lower()
    if not re.fullmatch(r"[0-9a-f]{64}", record_hash):
        raise RuntimeError("portable mapping row lacks source_record_sha256")
    evidence = None
    for line in source_record.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        if str(candidate.get("record_sha256") or "").lower() == record_hash:
            evidence = candidate
            break
    if not isinstance(evidence, dict):
        raise RuntimeError("portable mapping evidence record is absent")
    payload = dict(evidence)
    claimed = str(payload.pop("record_sha256", "")).lower()
    if _canonical_record_sha256(payload) != claimed:
        raise RuntimeError("portable mapping evidence record hash does not recompute")
    raw = evidence.get("raw_event") if isinstance(evidence.get("raw_event"), dict) else {}
    anchor = evidence.get("anchor") if isinstance(evidence.get("anchor"), dict) else {}
    expected = {
        "event_id": raw.get("graph_event_id") or raw.get("event_id"),
        "actor": raw.get("actorID"),
        "object": raw.get("objectID"),
        "action": raw.get("action"),
        "timestamp": raw.get("timestamp"),
        "source_event_sha256": raw.get("source_record_sha256"),
        "anchor": anchor.get("node_uuid"),
    }
    if raw.get("raw_event_uuid") not in (None, ""):
        expected["raw_event_uuid"] = raw.get("raw_event_uuid")
    if any(str(boundary.get(key) or "") != str(value or "") for key, value in expected.items()):
        raise RuntimeError("portable mapping boundary contradicts its source evidence")


def _iso_timestamp_seconds(value) -> float:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).timestamp()


def _canonical_graph_event_id(value) -> str:
    """Normalize the CDM UUID while preserving ATHENA's endpoint suffix."""
    text = str(value or "").strip()
    raw_uuid, separator, endpoint = text.partition(":")
    normalized = normalize_cdm_uuid(raw_uuid)
    return normalized + (separator + endpoint if separator else "")


def _resolve_mapping_registry(path: Path, handler, dataset: str, requested_scene: str | None) -> list[dict]:
    """Resolve Table VII anchors against the exact current audit event graph."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resolved = []
    keys = set()
    for row in rows:
        if row.get("record_type") != "mapping" or row.get("dataset") != dataset:
            continue
        declared_scene = str(row.get("scene") or "")
        resolve_scene = bool(
            declared_scene == "*"
            and row.get("scene_resolution") == "runtime_unique_raw_event"
        )
        if requested_scene is not None and not resolve_scene and declared_scene != requested_scene:
            continue
        source_record = Path(str(row.get("source_record") or ""))
        if not source_record.is_absolute():
            source_record = path.parent / source_record
        boundary = row.get("boundary") or {}
        scene = declared_scene
        host = str(row.get("host_id") or row.get("host") or "")
        anchor = str(row.get("anchor_uuid") or boundary.get("anchor") or "")
        portable_keys = (
            "event_id", "anchor", "actor", "object", "action", "timestamp",
            "source_event_sha256",
        )
        portable = boundary.get("snapshot") in (None, "")
        if (
            not source_record.is_file()
            or hashlib.sha256(source_record.read_bytes()).hexdigest() != row.get("source_hash")
            or not all((scene, host, anchor))
            or (scene == "*" and not resolve_scene)
            or (
                any(boundary.get(key) in (None, "") for key in portable_keys)
                if portable else
                any(boundary.get(key) in (None, "") for key in ("snapshot", "event_id", "anchor", "event_sha256"))
            )
        ):
            raise RuntimeError("mapping registry row lacks source provenance or exact event boundary")
        if portable:
            _portable_boundary_evidence(source_record, row, boundary)
            snapshot_candidates = range(len(handler.snapshots))
        else:
            snapshot_id = int(boundary["snapshot"])
            if snapshot_id < 0 or snapshot_id >= len(handler.snapshots):
                raise RuntimeError("mapping registry snapshot is outside the interpreted graph stream")
            snapshot_candidates = (snapshot_id,)
        if resolve_scene and boundary.get("raw_event_uuid") in (None, ""):
            raise RuntimeError("portable E5 mapping lacks raw_event_uuid")
        matches = []
        for snapshot_id in snapshot_candidates:
            graph = handler.snapshots[snapshot_id]
            actual_scene = _snapshot_source_scene(graph, requested_scene)
            if (
                (not resolve_scene and actual_scene != scene)
                or normalize_cdm_uuid(_snapshot_host_id(graph)) != normalize_cdm_uuid(host)
            ):
                continue
            for edge_index, edge in enumerate(graph.es):
                attrs = edge.attributes()
                if _canonical_graph_event_id(attrs.get("event_id")) != _canonical_graph_event_id(boundary["event_id"]):
                    continue
                source = str(graph.vs[edge.source]["name"])
                target = str(graph.vs[edge.target]["name"])
                canonical_anchor = normalize_cdm_uuid(anchor)
                if (
                    canonical_anchor not in {normalize_cdm_uuid(source), normalize_cdm_uuid(target)}
                    or normalize_cdm_uuid(boundary["anchor"]) != canonical_anchor
                ):
                    continue
                if portable and (
                    normalize_cdm_uuid(source) != normalize_cdm_uuid(boundary["actor"])
                    or normalize_cdm_uuid(target) != normalize_cdm_uuid(boundary["object"])
                    or str(attrs.get("actions") or "") != str(boundary["action"])
                    or abs(float(attrs.get("timestamp", 0.0) or 0.0) - _iso_timestamp_seconds(boundary["timestamp"])) > 0.001
                ):
                    continue
                matches.append((snapshot_id, edge_index, graph, actual_scene))
        if len(matches) != 1:
            raise RuntimeError("mapping registry boundary does not uniquely join the audit event")
        snapshot_id, edge_index, graph, actual_scene = matches[0]
        resolved_event_hash = event_sha256(graph, edge_index)
        if not portable and resolved_event_hash != str(boundary["event_sha256"]):
            raise RuntimeError("mapping registry boundary does not join the hashed audit event")
        resolved_scene = actual_scene if resolve_scene else scene
        key = (dataset, resolved_scene, host, anchor, snapshot_id, str(boundary["event_id"]))
        if key in keys:
            raise RuntimeError("mapping registry has duplicate dataset/scene/host/anchor keys")
        keys.add(key)
        resolved.append({
            **row,
            "scene": resolved_scene,
            "host_id": host,
            "anchor_uuid": anchor,
            "source_record": str(source_record.resolve()),
            "boundary": {
                **boundary,
                "snapshot": snapshot_id,
                "edge_index": edge_index,
                "event_sha256": resolved_event_hash,
            },
        })
    return resolved


def _serialize_causal_paths(paths) -> list:
    """Keep the complete event-level trace used to form every ATT&CK query."""
    serialized = []
    for path in paths:
        trace = []
        for edge in path:
            src, dst, action = edge[:3]
            trace.append({
                "source": int(src),
                "target": int(dst),
                "action": str(action),
                "event_id": str(edge[3]) if len(edge) > 3 else "",
                "event_order": int(edge[4]) if len(edge) > 4 else -1,
            })
        serialized.append(trace)
    return serialized


class HostTacticQueues:
    """Independent persistent tactic queues with online LCS decisions."""

    def __init__(self, retention_seconds: float, top_k: int, min_ratio: float, library):
        self.retention_seconds = float(retention_seconds)
        self.top_k = int(top_k)
        self.min_ratio = float(min_ratio)
        self.library = library
        self.queues = {}

    def append(self, host_id: str, entry: dict) -> dict:
        host_id = str(host_id or "").strip()
        if not host_id:
            raise RuntimeError("cannot append to a persistent queue without host_id")
        queue = self.queues.setdefault(host_id, deque())
        queue.append(entry)
        now = float(entry["timestamp"])
        while queue and now - float(queue[0]["timestamp"]) > self.retention_seconds:
            queue.popleft()
        rows = list(queue)
        chains = build_candidate_tactic_chains(rows, top_k=self.top_k)
        aligned = align_candidate_chains(
            chains,
            self.library,
            min_ratio=self.min_ratio,
            top_k=self.top_k,
        )
        return {
            "host_id": host_id,
            "queue_size": len(rows),
            "candidate_tactic_chains": chains,
            "aligned_top_k_chains": aligned,
            "passes_lcs_filter": any(row["passes_threshold"] for row in aligned),
        }

    def rows(self) -> dict:
        return {host: list(queue) for host, queue in sorted(self.queues.items())}


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    interp_cfg = cfg.get("interpretation", {})
    top_k = args.top_k or int(interp_cfg.get("topk_candidates", 10))
    gamma = float(interp_cfg.get("gamma", 0.50))
    min_ratio = float(interp_cfg.get("lcs_min_ratio", 0.60))
    retention_days = int(interp_cfg.get("tactic_queue_retention_days", 7))
    fallback_hops = int(interp_cfg.get("lambda_hops", 4))
    max_paths_per_peer = int(interp_cfg.get("max_causal_paths_per_peer", 1024))
    max_paths_per_alert = int(interp_cfg.get("max_causal_paths_per_alert", 4096))
    max_path_expansions = int(interp_cfg.get("max_causal_path_expansions", 100_000))

    if not args.use_ground_truth and not args.detections:
        raise SystemExit(
            "run_interpretation.py now requires --detections <run_detection.json>. "
            "Use --use-ground-truth only for annotation/debug evaluation."
        )

    handler = get_handler(args.dataset, True, cfg.get("paths", {}), scene_name=args.scene)
    handler.load()
    handler.build_graph(args.dataset)
    interval_replay_meta = None
    detection_checkpoint = None
    attack_predictions = []
    registry_path = args.attack_event_boundaries
    if args.detections:
        detection_payload = json.loads(Path(args.detections).read_text(encoding="utf-8"))
        detection_checkpoint = detection_payload.get("checkpoint")
    if args.benign_injection_manifest:
        replay_source = json.loads(args.benign_injection_manifest.read_text(encoding="utf-8"))
        registry_path = Path(str((replay_source.get("attack_event_boundaries") or {}).get("path") or ""))
        attack_predictions = list(replay_source.get("attack_predictions") or [])
        if not attack_predictions:
            raise RuntimeError("event-level replay manifest contains no source-linked attack predictions")
        _train_ids, _test_ids, replay_base_split = build_split(
            handler, float(cfg.get("detection", {}).get("train_ratio", 0.70)),
        )
        _inserted, interval_replay_meta = apply_interval_replay(
            handler, args.benign_injection_manifest, args.dataset, args.scene,
            expected_split=replay_base_split,
        )
        if args.detections:
            if detection_payload.get("interval_replay") != interval_replay_meta:
                raise RuntimeError("detection and interpretation interval replay manifests differ")
    elif args.attack_event_boundaries:
        if not args.detections:
            raise RuntimeError("attack-event boundaries require a bound detection artifact")
        from scripts.build_benign_injection_manifest import _load_boundaries
        test_ids = [int(value) for value in (detection_payload.get("split") or {}).get("test_snapshots", [])]
        attack_events = _load_boundaries(
            args.attack_event_boundaries, handler, args.dataset, args.scene, test_ids,
            require_rows=False,
        )
        attack_predictions = [row for row in attack_events if row.get("prediction_id")]

    prediction_registry = {}
    for prediction in attack_predictions:
        ref = prediction.get("boundary") or {}
        event_id = str(ref.get("event_id") or "")
        matches = []
        for snapshot_id, graph in enumerate(handler.snapshots):
            if str(graph["source_scene"] or "") != str(prediction.get("scene") or "") or str(graph["host_id"] or "") != str(prediction.get("host") or ""):
                continue
            for edge_index, edge in enumerate(graph.es):
                if str(edge.attributes().get("event_id") or "") == event_id:
                    try:
                        raw_order = edge.attributes().get("event_order", edge_index)
                        event_order = int(edge_index if raw_order in (None, "") else raw_order)
                    except (TypeError, ValueError):
                        event_order = edge_index
                    matches.append((
                        float(edge.attributes().get("timestamp", 0.0) or 0.0),
                        event_order, snapshot_id, edge_index,
                    ))
        if len(matches) != 1:
            raise RuntimeError("attack prediction boundary does not uniquely join the interpreted event stream")
        key = (args.dataset, str(prediction["scene"]), str(prediction["host"]), str(prediction["prediction_id"]))
        if key in prediction_registry:
            raise RuntimeError("duplicate source-linked attack prediction key")
        prediction_registry[key] = {"prediction": prediction, "position": matches[0]}

    mapping_registry = (
        _resolve_mapping_registry(registry_path, handler, args.dataset, args.scene)
        if registry_path and not args.benign_injection_manifest else []
    )

    malicious_uuids = set()
    detected_by_snapshot = {}
    if args.use_ground_truth:
        mal_start = int(getattr(handler, "malicious_idx_start", -1) or -1)
        mal_end = int(getattr(handler, "malicious_idx_end", -1) or -1)
        if mal_start < 0 or mal_end < mal_start:
            print(f"[interpretation] no malicious snapshots in {args.dataset}/{args.scene}")
            return
        mal_indices = list(range(mal_start, mal_end + 1))
        malicious_uuids = load_malicious_uuids(args.dataset, args.scene or "")
        malicious_uuids.update(str(value) for value in getattr(handler, "all_labels", []))
        if not malicious_uuids:
            raise RuntimeError(
                f"no released malicious-entity labels found for dataset={args.dataset} scene={args.scene}"
            )
    else:
        detected_by_snapshot = _load_detected_nodes(
            args.detections,
            include_train=bool(args.include_train_detections),
            expected_dataset=args.dataset,
            expected_scene=args.scene,
        )
        mal_indices = sorted(detected_by_snapshot)
        if not mal_indices and not attack_predictions and not mapping_registry:
            print(f"[interpretation] detector produced no malicious nodes in {args.detections}")
            return

    mal_indices = sorted(
        mal_indices,
        key=lambda snapshot_id: (_snapshot_time_seconds(handler.snapshots[snapshot_id], snapshot_id), snapshot_id),
    )
    if args.max_malicious is not None:
        mal_indices = mal_indices[: args.max_malicious]

    marked_total = 0
    for sidx in mal_indices:
        if args.use_ground_truth:
            marked_total += _mark_malicious(handler.snapshots[sidx], malicious_uuids)
        else:
            marked_total += _mark_detected(handler.snapshots[sidx], detected_by_snapshot.get(sidx, set()))

    technique_file, log_enhanced = _mapping_variant_settings(args.mapping_variant)
    triples_path = str(
        REPO_ROOT / "data" / "attack_knowledge" / "mitre_attack" / technique_file
    )
    mapper = TechniqueSemanticMapper(
        triples_path=triples_path,
        model_name=str(interp_cfg.get("sentence_bert_model", "all-MiniLM-L12-v2")),
        threshold=gamma,
        top_k=top_k,
    )

    sequence_path = str(interp_cfg.get("attack_sequence_records", "") or "").strip()
    if sequence_path.startswith("<") and sequence_path.endswith(">"):
        raise RuntimeError(
            "configure interpretation.attack_sequence_records with a provenance-bearing "
            "AttackSeqBench-derived JSONL export"
        )
    sequence_file = Path(sequence_path)
    if not sequence_file.is_absolute():
        sequence_file = REPO_ROOT / sequence_file
    tactic_lib = load_tactic_sequence_records(str(sequence_file))
    if not tactic_lib:
        raise RuntimeError(
            f"no verified attack-sequence records loaded from {sequence_file}; "
            "each JSONL row must include source_id, source_record, source_hash, "
            "source_corpus, and techniques"
        )
    tech_to_tactics = load_tech_to_tactics()

    print(
        f"[interpretation] dataset={args.dataset} scene={args.scene} "
        f"malicious_snapshots={len(mal_indices)} mapper_top_k={top_k} "
        f"gamma={gamma} tactic_queue_retention_days={retention_days} "
        f"lcs_min_ratio={min_ratio} tactic_lib_size={len(tactic_lib)} "
        f"input={'ground_truth' if args.use_ground_truth else args.detections} "
        f"labels_loaded={len(malicious_uuids)} nodes_marked={marked_total}"
    )

    per_subgraph = []
    retention_seconds = float(retention_days) * 24 * 60 * 60
    host_queues = HostTacticQueues(retention_seconds, top_k, min_ratio, tactic_lib)
    final_detection_decisions = []
    prediction_online_alignments = {}
    ordered_positives = []
    for sidx in mal_indices:
        snap = handler.snapshots[sidx]
        host_id = _snapshot_host_id(snap)
        source_scene = _snapshot_source_scene(snap, args.scene)
        mal_nodes = _malicious_nodes(snap, malicious_uuids if args.use_ground_truth else set())
        for malicious_node in mal_nodes:
            node_uuid = str(snap.vs[malicious_node].attributes().get("name", ""))
            incident_event_ids = {
                str(snap.es[edge_id].attributes().get("event_id") or "")
                for edge_id in snap.incident(malicious_node, mode="ALL")
            }
            matches = [
                (key, value) for key, value in prediction_registry.items()
                if key[1] == source_scene and key[2] == host_id
                and str(value["prediction"]["boundary"]["anchor"]) == node_uuid
                and str(value["prediction"]["boundary"]["event_id"]) in incident_event_ids
            ]
            if len(matches) > 1:
                raise RuntimeError("detector positive ambiguously joins multiple source-linked attack cases")
            prediction_key = matches[0][0] if matches else None
            commit_order = _prediction_commit_order(
                snap, malicious_node, matches[0][1]["position"] if matches else None,
            )
            ordered_positives.append({
                "snapshot": sidx,
                "node": malicious_node,
                "nodes": mal_nodes,
                "host_id": host_id,
                "source_scene": source_scene,
                "prediction_key": prediction_key,
                "commit_order": commit_order,
                "incident_event_ids": incident_event_ids,
            })

    ordered_positives.sort(key=lambda row: (
        row["host_id"], row["commit_order"], row["source_scene"], row["snapshot"], row["node"],
    ))
    for work in ordered_positives:
        sidx = work["snapshot"]
        snap = handler.snapshots[sidx]
        host_id, source_scene = work["host_id"], work["source_scene"]
        mal_nodes = work["nodes"]
        snap_time = _snapshot_time_seconds(snap, sidx)
        for malicious_node in [work["node"]]:
            node_uuid = str(snap.vs[malicious_node].attributes().get("name", ""))
            incident_event_ids = work["incident_event_ids"]
            prediction_key = work["prediction_key"]
            attack_id = prediction_key[3] if prediction_key else None
            decision_base = {
                "snapshot": sidx,
                "snapshot_time": snap_time,
                "host_id": host_id,
                "source_scene": source_scene,
                "attack_id": attack_id,
                "malicious_node": malicious_node,
                "uuid": node_uuid,
                "detector_positive": True,
                "incident_event_ids": sorted(incident_event_ids),
            }
            paths, path_audit = reconstruct_attack_paths(
                snap,
                malicious_node,
                mal_nodes,
                fallback_hops=fallback_hops,
                max_paths_per_peer=max_paths_per_peer,
                max_paths_per_alert=max_paths_per_alert,
                max_expansions_per_alert=max_path_expansions,
                return_audit=True,
            )
            path_queries = [
                path_edges_to_query(snap, path, enhance_identifiers=log_enhanced)
                for path in paths
            ]
            path_queries = [query for query in path_queries if query]
            if not path_queries:
                final_detection_decisions.append({
                    **decision_base,
                    "final_alert": False,
                    "filtered_as_false_positive": True,
                    "filter_reason": "no_causal_path_query",
                    "lcs_alignment": None,
                    "causal_path_audit": path_audit,
                })
                per_subgraph.append({
                    "snapshot": sidx, "snapshot_time": snap_time,
                    "malicious_node": malicious_node, "uuid": decision_base["uuid"],
                    "host_id": host_id, "source_scene": source_scene,
                    "attack_id": attack_id, "causal_paths": 0, "path_edges": 0,
                    "causal_path_traces": [], "causal_path_audit": path_audit,
                    "top_k_candidates": [], "path_query_previews": [],
                    "final_alert": False, "unmapped_reason": "no_causal_path_query",
                    "max_similarity": None, "gamma": gamma,
                    "incident_event_ids": sorted(incident_event_ids),
                })
                continue

            by_technique = {}
            max_similarity = float("-inf")
            missing_tactic = set()
            for query in path_queries:
                for candidate in mapper.predict_top_k_detail(query):
                    tech_id = normalize_tech_id(str(candidate.get("tech_id", "")))
                    score = float(candidate.get("score", 0.0))
                    max_similarity = max(max_similarity, score)
                    if not tech_id or score < gamma:
                        continue
                    if not tech_to_tactics.get(tech_id):
                        missing_tactic.add(tech_id)
                        continue
                    previous = by_technique.get(tech_id)
                    if previous is None or score > float(previous.get("score", 0.0)):
                        by_technique[tech_id] = {
                            "technique": tech_id,
                            "tactics": tech_to_tactics[tech_id],
                            "score": score,
                        }
            if missing_tactic:
                raise RuntimeError(
                    "ATT&CK tactic mapping is missing for above-gamma techniques: "
                    + ", ".join(sorted(missing_tactic))
                )
            candidates = sorted(
                by_technique.values(),
                key=lambda row: (-row["score"], row["technique"]),
            )[:top_k]
            if not candidates:
                final_detection_decisions.append({
                    **decision_base,
                    "final_alert": False,
                    "filtered_as_false_positive": True,
                    "filter_reason": "top_similarity_below_gamma",
                    "lcs_alignment": None,
                    "causal_path_audit": path_audit,
                })
                per_subgraph.append({
                    "snapshot": sidx, "snapshot_time": snap_time,
                    "malicious_node": malicious_node, "uuid": decision_base["uuid"],
                    "host_id": host_id, "source_scene": source_scene,
                    "attack_id": attack_id, "causal_paths": len(paths),
                    "path_edges": sum(len(path) for path in paths),
                    "causal_path_traces": _serialize_causal_paths(paths),
                    "causal_path_audit": path_audit, "top_k_candidates": [],
                    "path_query_previews": [query[:160] for query in path_queries],
                    "final_alert": False,
                    "unmapped_reason": "top_similarity_below_gamma",
                    "max_similarity": max_similarity, "gamma": gamma,
                    "incident_event_ids": sorted(incident_event_ids),
                })
                continue

            entry = {
                "timestamp": float(work["commit_order"][0]),
                "event_order": int(work["commit_order"][1]),
                "host_id": host_id,
                "source_scene": source_scene,
                "attack_id": attack_id,
                "snapshot": sidx,
                "malicious_node": malicious_node,
                "attack_entry_id": (
                    f"{args.dataset}:{source_scene}:{host_id}:{sidx}:{malicious_node}"
                ),
                "technique_set": [candidate["technique"] for candidate in candidates],
                "tactic_set": list(dict.fromkeys(
                    tactic for candidate in candidates for tactic in candidate["tactics"]
                )),
                "candidates": candidates,
            }
            online_alignment = host_queues.append(host_id, entry)
            if prediction_key in prediction_registry:
                prediction_online_alignments[prediction_key] = online_alignment
            decision = {
                **decision_base,
                "final_alert": bool(online_alignment["passes_lcs_filter"]),
                "filtered_as_false_positive": not bool(online_alignment["passes_lcs_filter"]),
                "filter_reason": (
                    None if online_alignment["passes_lcs_filter"] else "lcs_below_threshold"
                ),
                "lcs_alignment": online_alignment,
                "causal_path_audit": path_audit,
            }
            final_detection_decisions.append(decision)
            per_subgraph.append({
                "snapshot": sidx,
                "snapshot_time": snap_time,
                "malicious_node": malicious_node,
                "uuid": decision_base["uuid"],
                "host_id": host_id,
                "source_scene": source_scene,
                "attack_id": attack_id,
                "causal_paths": len(paths),
                "path_edges": sum(len(path) for path in paths),
                "causal_path_traces": _serialize_causal_paths(paths),
                "causal_path_audit": path_audit,
                "top_k_candidates": candidates,
                "path_query_previews": [query[:160] for query in path_queries],
                "final_alert": decision["final_alert"],
                "max_similarity": max_similarity, "gamma": gamma,
                "incident_event_ids": sorted(incident_event_ids),
            })

    queue_rows_by_host = host_queues.rows()

    final_attack_predictions = []
    for prediction_key, value in sorted(prediction_registry.items()):
        prediction = value["prediction"]
        attack_id = str(prediction["prediction_id"])
        online = prediction_online_alignments.get(prediction_key)
        positive_count = sum(
            int(
                row.get("attack_id") == attack_id
                and row.get("source_scene") == str(prediction["scene"])
                and row.get("host_id") == str(prediction["host"])
            )
            for row in per_subgraph
        )
        final_attack_predictions.append({
            "dataset": args.dataset,
            "source_scene": str(prediction["scene"]),
            "host_id": str(prediction["host"]),
            "attack_id": attack_id,
            "candidate_tactic_chains": list((online or {}).get("candidate_tactic_chains", [])),
            "aligned_top_k_chains": list((online or {}).get("aligned_top_k_chains", [])),
            "detector_positive_count": positive_count,
            "empty_reason": (
                None if online else
                "detected_but_unmapped" if positive_count else "no_detector_positive_for_case"
            ),
        })

    final_mapping_predictions = []
    for expected in mapping_registry:
        boundary_event = str(expected["boundary"]["event_id"])
        matches = [row for row in per_subgraph
                   if int(row.get("snapshot", -1)) == int(expected["boundary"]["snapshot"])
                   and row.get("source_scene") == str(expected["scene"])
                   and row.get("host_id") == str(expected["host_id"])
                   and row.get("uuid") == str(expected["anchor_uuid"])
                   and boundary_event in row.get("incident_event_ids", [])]
        if len(matches) > 1:
            raise RuntimeError("mapping registry anchor joins multiple runtime prediction rows")
        runtime = matches[0] if matches else None
        final_mapping_predictions.append({
            "source_id": str(expected.get("source_id") or ""),
            "source_scene": str(expected["scene"]), "host_id": str(expected["host_id"]),
            "anchor_uuid": str(expected["anchor_uuid"]),
            "boundary_snapshot": int(expected["boundary"]["snapshot"]),
            "boundary_event_id": boundary_event,
            "boundary_event_sha256": str(expected["boundary"]["event_sha256"]),
            "top_k_candidates": list((runtime or {}).get("top_k_candidates", [])),
            "unmapped_reason": ((runtime or {}).get("unmapped_reason") if runtime else "detector_miss"),
            "max_similarity": (runtime or {}).get("max_similarity"), "gamma": gamma,
        })

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "scene": args.scene,
            "mapping_variant": args.mapping_variant,
            "persistent_tactic_queues_by_host": queue_rows_by_host,
            "final_attack_predictions": final_attack_predictions,
            "final_mapping_predictions": final_mapping_predictions,
            "final_detection_decisions": final_detection_decisions,
            "final_alerts": [row for row in final_detection_decisions if row["final_alert"]],
            "filtered_detector_positives": [
                row for row in final_detection_decisions if row["filtered_as_false_positive"]
            ],
            "lcs_min_ratio": min_ratio,
            "tactic_queue_retention_days": retention_days,
            "input_mode": "ground_truth" if args.use_ground_truth else "detector_output",
            "detections": args.detections,
            "interval_replay": interval_replay_meta,
            "attack_event_boundaries": (
                {
                    "path": str(args.attack_event_boundaries.resolve()),
                    "sha256": hashlib.sha256(args.attack_event_boundaries.read_bytes()).hexdigest(),
                }
                if args.attack_event_boundaries else
                (json.loads(args.benign_injection_manifest.read_text(encoding="utf-8")).get("attack_event_boundaries")
                 if args.benign_injection_manifest else None)
            ),
            "detection_checkpoint": detection_checkpoint,
            "per_attack_subgraph": per_subgraph,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[interpretation] wrote {out_path}")


if __name__ == "__main__":
    main()
