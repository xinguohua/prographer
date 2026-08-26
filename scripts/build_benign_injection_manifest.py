"""Build a source-linked, event-boundary E3 replay manifest for Table VIII."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_detection import prepare_data
from src.utils.config import load_config
from src.utils.interval_replay import (
    _event_dataframe,
    canonical_event_row_payload,
    canonical_event_row_sha256,
    event_sha256,
    graph_sha256,
    validate_source_event_window,
)
from src.utils.split import build_split

E3_DATASETS = {"cadets", "theia", "trace", "clearscope"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _time(graph, fallback: float) -> float:
    if "window_start" in graph.attributes():
        return float(graph["window_start"])
    values = []
    for edge in graph.es:
        try:
            value = float(edge.attributes().get("timestamp", 0.0) or 0.0)
            scale = 1e9 if abs(value) > 1e17 else 1e6 if abs(value) > 1e14 else 1e3 if abs(value) > 1e11 else 1.0
            values.append(value / scale)
        except (TypeError, ValueError):
            pass
    return min(values) if values else fallback


def _event_time(graph, edge_index: int) -> float:
    value = float(graph.es[int(edge_index)].attributes().get("timestamp", 0.0) or 0.0)
    scale = 1e9 if abs(value) > 1e17 else 1e6 if abs(value) > 1e14 else 1e3 if abs(value) > 1e11 else 1.0
    return value / scale


def _raw_event_times(handler) -> dict[str, int]:
    frames = [frame for frame in (getattr(handler, "begin", None), getattr(handler, "malicious", None))
              if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        raise ValueError("E3 loader did not retain source event rows")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    event_ids = frame["event_id"].astype(str)
    duplicated = sorted(set(event_ids[event_ids.duplicated(keep=False)].tolist()))
    if duplicated:
        raise ValueError(f"retained E3 event_id values are not one-to-one: {duplicated[:5]}")
    return {str(row.event_id): int(str(row.timestamp)) for row in frame.itertuples(index=False)}


def _benign(graph) -> bool:
    return "label" in graph.vs.attributes() and all(int(value or 0) == 0 for value in graph.vs["label"])


def _resolve_source(boundary_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else boundary_path.parent / path


def _load_boundaries(
    path: Path, handler, dataset: str, scene: str | None, test_ids: list[int],
    *, require_rows: bool = True,
) -> list[dict]:
    all_rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in all_rows
            if row.get("dataset") == dataset and row.get("record_type") in {"sequence", "attack_event"}]
    if not rows and require_rows:
        raise ValueError("attack-event boundary JSONL is empty")
    event_index = {}
    for snapshot_id in test_ids:
        graph = handler.snapshots[snapshot_id]
        for edge_index, edge in enumerate(graph.es):
            event_id = str(edge.attributes().get("event_id") or f"snapshot-{snapshot_id}-edge-{edge_index}")
            for anchor in {str(graph.vs[edge.source]["name"]), str(graph.vs[edge.target]["name"])}:
                event_index.setdefault((snapshot_id, event_id, anchor), []).append(edge_index)
    raw_times = _raw_event_times(handler)
    resolved = []
    for row in rows:
        record_type = str(row.get("record_type"))
        prediction_id = str(row.get("prediction_id") or row.get("attack_id") or "") if record_type == "sequence" else None
        attack_event_id = str(row.get("attack_event_id") or (f"prediction:{prediction_id}" if prediction_id else ""))
        host = str(row.get("host_id") or row.get("host") or "")
        source_record = _resolve_source(path, str(row.get("source_record") or ""))
        ref = row.get("boundary")
        if (
            row.get("dataset") != dataset
            or not row.get("scene")
            or (scene is not None and row.get("scene") != scene)
            or not row.get("source_id") or not row.get("source_corpus")
            or not source_record.is_file() or _sha256(source_record) != row.get("source_hash")
            or not host or not attack_event_id or (record_type == "sequence" and not prediction_id)
            or not isinstance(ref, dict)
        ):
            raise ValueError("attack-event prediction row lacks source provenance or boundary")
        snapshot_id = int(ref.get("snapshot", -1))
        event_id, anchor = str(ref.get("event_id") or ""), str(ref.get("anchor") or "")
        matches = event_index.get((snapshot_id, event_id, anchor), [])
        if len(matches) != 1:
            raise ValueError("attack-event boundary does not uniquely join snapshot/event_id/anchor")
        edge_index = matches[0]
        graph = handler.snapshots[snapshot_id]
        edge = graph.es[edge_index]
        attack_endpoints = {str(graph.vs[index]["name"]) for index in (edge.source, edge.target)
                            if int(graph.vs[index]["label"] or 0) == 1}
        if anchor not in attack_endpoints:
            raise ValueError("attack-event boundary anchor is not a malicious event endpoint")
        if str(graph["host_id"] or "") != host or str(graph["source_scene"] or "") != str(row.get("scene") or ""):
            raise ValueError("attack-event boundary host/scene contradicts the audit event")
        digest = event_sha256(graph, edge_index)
        if ref.get("event_sha256") != digest or event_id not in raw_times:
            raise ValueError("attack-event boundary hash/raw audit join mismatch")
        checked = {"snapshot": snapshot_id, "edge_index": edge_index, "event_id": event_id,
                   "anchor": anchor, "event_sha256": digest, "event_time": raw_times[event_id]}
        resolved.append({"source_id": row["source_id"], "source_corpus": row["source_corpus"],
                         "source_record": str(source_record.resolve()), "source_hash": row["source_hash"],
                         "record_type": record_type, "dataset": dataset, "scene": str(row["scene"]), "host": host,
                         "attack_event_id": attack_event_id, "prediction_id": prediction_id,
                         "boundary": checked})
    attack_event_ids = [row["attack_event_id"] for row in resolved]
    if len(attack_event_ids) != len(set(attack_event_ids)):
        raise ValueError("attack_event_id must be unique within the selected dataset registry")
    return resolved


def _load_source_plan(path: Path, dataset: str, condition: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("benign source plan is empty")
    for row in rows:
        source_record = _resolve_source(path, str(row.get("source_record") or ""))
        start = int(row.get("source_start_timestamp", 0))
        end = int(row.get("source_end_timestamp", 0))
        minute_ns = 60 * 1_000_000_000
        if (
            row.get("dataset") != dataset or row.get("condition") != condition
            or not row.get("source_id") or not row.get("source_corpus")
            or not source_record.is_file() or _sha256(source_record) != row.get("source_hash")
            or not row.get("host") or not row.get("before_attack_event_id")
            or not row.get("after_attack_event_id") or row.get("reuse_policy") not in {"allow", "forbid"}
            or not isinstance(row.get("source_snapshots"), list) or not row["source_snapshots"]
            or not isinstance(row.get("source_snapshot_sha256"), list)
            or len(row["source_snapshots"]) != len(row["source_snapshot_sha256"])
            or end <= start or start % minute_ns or end % minute_ns
        ):
            raise ValueError("benign source plan row lacks provenance, gap key, or reuse policy")
        row["source_record"] = str(source_record.resolve())
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(E3_DATASETS))
    parser.add_argument("--scene")
    parser.add_argument("--condition", required=True, choices=("24h", "48h", "72h"))
    parser.add_argument("--attack-event-boundaries", required=True, type=Path)
    parser.add_argument("--benign-source-plan", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spec-output", required=True, type=Path)
    args = parser.parse_args(argv)
    if not args.attack_event_boundaries.is_file() or not args.benign_source_plan.is_file() or not args.checkpoint.is_file():
        raise ValueError("boundary registry, benign source plan, and frozen Basic E3 checkpoint must exist")
    cfg = load_config(args.config)
    handler = prepare_data(cfg.get("paths", {}), args.dataset, args.scene)
    train_list, test_ids, split = build_split(handler, float(cfg.get("detection", {}).get("train_ratio", 0.70)))
    train_ids = set(train_list)
    attack_events = _load_boundaries(args.attack_event_boundaries, handler, args.dataset, args.scene, test_ids)
    attack_events.sort(key=lambda row: (
        str(row["host"]), int(row["boundary"]["event_time"]), int(row["boundary"]["snapshot"]),
        int(row["boundary"]["edge_index"]), str(row["attack_event_id"]),
    ))
    predictions = [row for row in attack_events if row.get("prediction_id")]
    prediction_keys = [
        (row["dataset"], row["scene"], row["host"], int(row["boundary"]["snapshot"]), row["boundary"]["anchor"])
        for row in predictions
    ]
    if len(prediction_keys) != len(set(prediction_keys)):
        raise ValueError("duplicate dataset/scene/host/prediction_id in boundary registry")
    seconds = int(args.condition[:-1]) * 3600.0
    source_plan = _load_source_plan(args.benign_source_plan, args.dataset, args.condition)
    plan_index = {
        (str(row["host"]), str(row["before_attack_event_id"]), str(row["after_attack_event_id"])): row
        for row in source_plan
    }
    if len(plan_index) != len(source_plan):
        raise ValueError("duplicate benign source-plan gap key")
    replay_gaps, inserted_events, consumed_plan_keys = [], [], set()
    used_slices = []
    by_stream = {}
    for row in attack_events:
        by_stream.setdefault(row["host"], []).append(row)
    for host, rows in sorted(by_stream.items()):
        for gap_index, (before_row, after_row) in enumerate(zip(rows, rows[1:])):
            plan_key = (host, before_row["attack_event_id"], after_row["attack_event_id"])
            plan = plan_index.get(plan_key)
            if plan is None:
                raise ValueError(f"benign source plan does not cover attack-event gap {plan_key}")
            consumed_plan_keys.add(plan_key)
            source_ids = [int(value) for value in plan["source_snapshots"]]
            hashes = list(plan["source_snapshot_sha256"])
            source_start_ns, source_end_ns = int(plan["source_start_timestamp"]), int(plan["source_end_timestamp"])
            actual_window = [
                index for index, graph in enumerate(handler.snapshots)
                if str(graph["host_id"] or "") == host
                and source_start_ns <= int(round(_time(graph, index * 60.0) * 1_000_000_000)) < source_end_ns
            ]
            if (
                source_end_ns - source_start_ns != int(seconds * 1_000_000_000)
                or source_ids != actual_window or len(hashes) != len(source_ids)
                or any(index not in train_ids or not _benign(handler.snapshots[index]) for index in source_ids)
                or hashes != [graph_sha256(handler.snapshots[index]) for index in source_ids]
            ):
                raise ValueError("benign source plan slice is not the complete hashed train-only benign window")
            overlaps = [previous for previous in used_slices
                        if previous["host"] == host and not (
                            source_end_ns <= previous["start"] or source_start_ns >= previous["end"]
                        )]
            if overlaps and (plan["reuse_policy"] != "allow" or any(row["policy"] != "allow" for row in overlaps)):
                raise ValueError("benign source plan reuses a slice without bilateral allow policy")
            used_slices.append({"host": host, "start": source_start_ns, "end": source_end_ns,
                                "policy": plan["reuse_policy"]})
            gap_ordinal = len(replay_gaps)
            replay_gaps.append({"attack_event_id": before_row["attack_event_id"], "gap_index": gap_index,
                                "gap_ordinal": gap_ordinal,
                                "source_scene": before_row["scene"], "host": host,
                                "next_source_scene": after_row["scene"],
                                "before": before_row["boundary"], "after": after_row["boundary"],
                                "next_attack_event_id": after_row["attack_event_id"], "interval_seconds": seconds,
                                "source_start_timestamp": source_start_ns,
                                "source_end_timestamp": source_end_ns,
                                "source_snapshots": source_ids,
                                "source_snapshot_scenes": [str(handler.snapshots[index]["source_scene"] or "") for index in source_ids],
                                "source_snapshot_sha256": hashes,
                                "source_plan_id": plan["source_id"], "reuse_policy": plan["reuse_policy"],
                                "source_slice_reused": bool(overlaps)})
            source_locations = {}
            for source_id in source_ids:
                graph = handler.snapshots[source_id]
                for edge_index, edge in enumerate(graph.es):
                    event_id = str(edge.attributes().get("event_id") or f"snapshot-{source_id}-edge-{edge_index}")
                    if event_id in source_locations:
                        raise ValueError("benign source plan event_id is not unique across its source window")
                    source_locations[event_id] = (source_id, f"snapshot:{source_id}/edge:{edge_index}")
            source_frame = _event_dataframe(handler, set(source_locations))
            validate_source_event_window(source_frame, source_start_ns, source_end_ns, host)
            for _, raw_row in source_frame.iterrows():
                event_id = str(raw_row["event_id"])
                source_id, locator = source_locations[event_id]
                payload = canonical_event_row_payload(raw_row)
                inserted_events.append({
                    "label": 0, "source_event_id": event_id,
                    "source_event_payload": payload,
                    "source_event_hash": canonical_event_row_sha256(raw_row),
                    "source_locator": locator,
                    "source_snapshot": source_id, "source_plan_id": plan["source_id"],
                    "attack_event_id": before_row["attack_event_id"],
                })
    if consumed_plan_keys != set(plan_index):
        raise ValueError("benign source plan contains gaps absent from the attack-event registry")
    if not replay_gaps:
        raise ValueError("boundary JSONL produced no consecutive attack-event gaps")
    manifest = {"schema_version": 2, "dataset": args.dataset, "scene": args.scene,
                "condition": args.condition, "config_sha256": _sha256(args.config),
                "base_split": split,
                "attack_event_boundaries": {"path": str(args.attack_event_boundaries.resolve()),
                                             "sha256": _sha256(args.attack_event_boundaries), "records": len(attack_events)},
                "benign_source_plan": {"path": str(args.benign_source_plan.resolve()),
                                       "sha256": _sha256(args.benign_source_plan), "records": len(source_plan)},
                "attack_events": attack_events, "attack_predictions": predictions,
                "replay_gaps": replay_gaps,
                "inserted_benign_events": inserted_events,
                "source_snapshot_reuse": any(gap["source_slice_reused"] for gap in replay_gaps)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    spec = {"source_id": f"{args.dataset}-{args.scene}-{args.condition}",
            "source_corpus": "ATHENA-E3-event-boundary-replay", "dataset": args.dataset,
            "scene": args.scene, "condition": args.condition, "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()), "source_event_manifest": str(args.output.resolve()),
            "attack_event_boundaries": str(args.attack_event_boundaries.resolve()),
            "benign_source_plan": str(args.benign_source_plan.resolve())}
    args.spec_output.parent.mkdir(parents=True, exist_ok=True)
    args.spec_output.write_text(json.dumps(spec, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "spec": str(args.spec_output), "gaps": len(replay_gaps)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
