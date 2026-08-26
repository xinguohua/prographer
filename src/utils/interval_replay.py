"""Deterministic source-linked benign-window replay for Table VIII."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


def _canonical_value(value):
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_canonical_value(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def canonical_event_row_payload(row) -> dict:
    """Canonical retained raw audit row used to bind each replayed event."""
    items = row.items() if hasattr(row, "items") else vars(row).items()
    return {
        str(key): _canonical_value(value)
        for key, value in sorted(items, key=lambda pair: str(pair[0]))
    }


def canonical_event_row_sha256(row) -> str:
    payload = canonical_event_row_payload(row)
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def graph_sha256(graph) -> str:
    nodes = [
        {
            key: vertex.attributes().get(key)
            for key in ("name", "type", "properties", "label")
        }
        for vertex in graph.vs
    ]
    nodes.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, default=str))
    edges = []
    for edge in graph.es:
        attrs = edge.attributes()
        edges.append({
            "source": str(graph.vs[edge.source]["name"]),
            "target": str(graph.vs[edge.target]["name"]),
            "actions": str(attrs.get("actions", "")),
            "event_id": str(attrs.get("event_id", "")),
            "timestamp": float(attrs.get("timestamp", 0.0) or 0.0),
        })
    payload = {"nodes": nodes, "edges": sorted(edges, key=lambda row: json.dumps(row, sort_keys=True))}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def event_sha256(graph, edge_index: int) -> str:
    """Hash one concrete audit event and its two snapshot-local endpoints."""
    edge = graph.es[int(edge_index)]
    payload = {
        "source": {
            key: graph.vs[edge.source].attributes().get(key)
            for key in ("name", "type", "properties", "label", "_athena_temporal_id")
        },
        "target": {
            key: graph.vs[edge.target].attributes().get(key)
            for key in ("name", "type", "properties", "label", "_athena_temporal_id")
        },
        "edge": edge.attributes(),
    }
    return hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def _event_ids(graph, snapshot_id: int) -> list[str]:
    return [
        str(edge.attributes().get("event_id") or f"snapshot-{snapshot_id}-edge-{index}")
        for index, edge in enumerate(graph.es)
    ]


def _event_dataframe(handler, event_ids: set[str]) -> pd.DataFrame:
    frames = [
        frame for frame in (getattr(handler, "begin", None), getattr(handler, "malicious", None))
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    if not frames or not event_ids:
        raise ValueError("interval replay requires the E3 event DataFrames retained by the loader")
    frame = pd.concat(frames, ignore_index=True, sort=False)
    if "event_id" not in frame.columns:
        raise ValueError("E3 event DataFrames do not retain event_id")
    selected = frame[frame["event_id"].astype(str).isin(event_ids)].copy()
    found = set(selected["event_id"].astype(str))
    if found != event_ids:
        missing = sorted(event_ids - found)[:5]
        raise ValueError(f"interval replay cannot join audit event_id values: {missing}")
    counts = selected["event_id"].astype(str).value_counts()
    if any(int(value) != 1 for value in counts) or len(selected) != len(event_ids):
        raise ValueError("interval replay requires event_id to join exactly one retained audit row")
    selected["_athena_input_order"] = range(len(selected))
    return selected


def _raw_timestamp(value) -> int:
    try:
        return int(Decimal(str(value)))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid E3 event timestamp: {value!r}") from exc


def validate_source_event_window(frame: pd.DataFrame, start: int, end: int, host: str) -> None:
    """Require every retained source event to lie in the exact planned host window."""
    timestamps = frame["timestamp"].map(_raw_timestamp)
    if (
        int(end) <= int(start)
        or not timestamps.map(lambda value: int(start) <= value < int(end)).all()
        or not frame["host_id"].astype(str).eq(str(host)).all()
    ):
        raise ValueError("benign source event lies outside its hashed host/time window")


def _replace_test_snapshots(handler, test_ids: list[int], replay_df: pd.DataFrame) -> list[int]:
    rebuilt = handler.create_snapshots_from_graph(replay_df.copy(), is_malicious=True)
    if len(rebuilt) < len(test_ids):
        raise ValueError("event-level replay unexpectedly removed held-out snapshot windows")
    for target_id, graph in zip(test_ids, rebuilt[:len(test_ids)]):
        handler.snapshots[target_id] = graph
    inserted_ids = []
    for graph in rebuilt[len(test_ids):]:
        handler.snapshots.append(graph)
        inserted_ids.append(len(handler.snapshots) - 1)
    return inserted_ids


def apply_interval_replay(
    handler, manifest_path: Path, dataset: str, scene: str | None,
    expected_split: dict | None = None,
) -> tuple[list[int], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", 0)) != 2:
        raise ValueError("Table VIII replay requires the event-boundary manifest schema v2")
    if manifest.get("dataset") != dataset or manifest.get("scene") != scene:
        raise ValueError("interval replay manifest dataset/scene mismatch")
    base_split = manifest.get("base_split") if isinstance(manifest.get("base_split"), dict) else {}
    if expected_split is not None and any(
        list(base_split.get(key, [])) != list(expected_split.get(key, []))
        for key in ("train_snapshots", "test_snapshots")
    ):
        raise ValueError("interval replay base split does not match the loaded E3 stream")
    train_ids = {int(value) for value in base_split.get("train_snapshots", [])}
    test_ids = [int(value) for value in base_split.get("test_snapshots", [])]
    test_id_set = set(test_ids)
    snapshots = handler.snapshots
    boundary = manifest.get("attack_event_boundaries") or {}
    boundary_path = Path(str(boundary.get("path") or ""))
    if not boundary_path.is_absolute():
        boundary_path = manifest_path.parent / boundary_path
    if not boundary_path.is_file() or hashlib.sha256(boundary_path.read_bytes()).hexdigest() != boundary.get("sha256"):
        raise ValueError("attack-event boundary source path/hash mismatch")

    plan_meta = manifest.get("benign_source_plan") or {}
    plan_path = Path(str(plan_meta.get("path") or ""))
    if not plan_path.is_absolute():
        plan_path = manifest_path.parent / plan_path
    if not plan_path.is_file() or hashlib.sha256(plan_path.read_bytes()).hexdigest() != plan_meta.get("sha256"):
        raise ValueError("benign source-plan path/hash mismatch")

    from scripts.build_benign_injection_manifest import _load_boundaries, _load_source_plan
    canonical_events = _load_boundaries(boundary_path, handler, dataset, scene, test_ids)
    canonical_events.sort(key=lambda row: (
        str(row["host"]), int(row["boundary"]["event_time"]),
        int(row["boundary"]["snapshot"]), int(row["boundary"]["edge_index"]), str(row["attack_event_id"]),
    ))
    if (
        canonical_events != manifest.get("attack_events")
        or [row for row in canonical_events if row.get("prediction_id")] != manifest.get("attack_predictions")
    ):
        raise ValueError("replay manifest events/predictions differ from the hashed boundary source")

    test_event_ids = {
        event_id for snapshot_id in test_ids for event_id in _event_ids(snapshots[snapshot_id], snapshot_id)
    }
    replay_df = _event_dataframe(handler, test_event_ids)
    original_ids = set(replay_df["event_id"].astype(str))
    gaps = manifest.get("replay_gaps")
    if not isinstance(gaps, list) or not gaps:
        raise ValueError("event-level replay manifest contains no verified attack-event gaps")
    expected_seconds = int(str(manifest["condition"]).removesuffix("h")) * 3600
    source_plan = _load_source_plan(plan_path, dataset, manifest["condition"])
    plan_index = {
        (str(row["host"]), str(row["before_attack_event_id"]), str(row["after_attack_event_id"])): row
        for row in source_plan
    }
    gap_plan_keys = {
        (str(gap.get("host") or ""), str(gap.get("attack_event_id") or ""),
         str(gap.get("next_attack_event_id") or ""))
        for gap in gaps
    }
    if set(plan_index) != gap_plan_keys or len(plan_index) != len(source_plan):
        raise ValueError("replay source-plan coverage differs from the canonical attack-event gaps")
    for gap in gaps:
        plan = plan_index.get((str(gap.get("host") or ""), str(gap.get("attack_event_id") or ""),
                               str(gap.get("next_attack_event_id") or "")))
        if plan is None or (
            str(gap.get("source_plan_id") or "") != str(plan["source_id"])
            or gap.get("reuse_policy") != plan.get("reuse_policy")
            or [int(value) for value in gap.get("source_snapshots", [])]
            != [int(value) for value in plan.get("source_snapshots", [])]
            or list(gap.get("source_snapshot_sha256", [])) != list(plan.get("source_snapshot_sha256", []))
            or int(gap.get("source_start_timestamp", 0)) != int(plan.get("source_start_timestamp", 0))
            or int(gap.get("source_end_timestamp", 0)) != int(plan.get("source_end_timestamp", 0))
        ):
            raise ValueError("derived replay gap differs from the hashed benign source plan")
    canonical_gaps = set()
    canonical_by_stream = {}
    for row in canonical_events:
        canonical_by_stream.setdefault(row["host"], []).append(row)
    for host, rows in canonical_by_stream.items():
        canonical_gaps.update(
            (host, before["scene"], after["scene"], before["attack_event_id"],
             after["attack_event_id"], before["boundary"]["event_id"], after["boundary"]["event_id"])
            for before, after in zip(rows, rows[1:])
        )
    actual_gaps = {
        (
            str(gap.get("host") or ""), str(gap.get("source_scene") or ""),
            str(gap.get("next_source_scene") or ""), str(gap.get("attack_event_id") or ""),
            str(gap.get("next_attack_event_id") or ""), str((gap.get("before") or {}).get("event_id") or ""),
            str((gap.get("after") or {}).get("event_id") or ""),
        )
        for gap in gaps
    }
    if actual_gaps != canonical_gaps or len(actual_gaps) != len(gaps):
        raise ValueError("replay gaps do not exactly cover canonical adjacent attack boundaries")
    used_source_ids, source_ranges, previous_before, any_reuse = set(), {}, {}, False
    for gap in gaps:
        stream = str(gap.get("host") or "")
        source_ids = [int(value) for value in gap.get("source_snapshots", [])]
        hashes = list(gap.get("source_snapshot_sha256", []))
        start, end = int(gap.get("source_start_timestamp", 0)), int(gap.get("source_end_timestamp", 0))
        before_time = int((gap.get("before") or {}).get("event_time", 0))
        if (
            int(gap.get("interval_seconds", 0)) != expected_seconds
            or end - start != expected_seconds * 1_000_000_000
            or not source_ids or len(source_ids) != len(hashes)
            or before_time < previous_before.get(stream, -1)
        ):
            raise ValueError("event-level replay gap violates duration, order, or source-plan contract")
        overlaps = [row for row in source_ranges.get(stream, [])
                    if not (end <= row[0] or start >= row[1]) or set(source_ids).intersection(row[3])]
        actual_reuse = bool(overlaps)
        if (
            actual_reuse != bool(gap.get("source_slice_reused"))
            or actual_reuse and (
                gap.get("reuse_policy") != "allow" or any(row[2] != "allow" for row in overlaps)
            )
        ):
            raise ValueError("event-level replay source reuse contradicts the verified source plan")
        any_reuse = any_reuse or actual_reuse
        used_source_ids.update(source_ids)
        source_ranges.setdefault(stream, []).append((start, end, gap.get("reuse_policy"), set(source_ids)))
        previous_before[stream] = before_time
    if bool(manifest.get("source_snapshot_reuse")) != any_reuse:
        raise ValueError("event-level replay aggregate source-reuse flag is inconsistent")

    event_order_by_stream = {}
    timed = replay_df.assign(_ts=replay_df["timestamp"].map(_raw_timestamp))
    for stream, part in timed.groupby("host_id", sort=False):
        event_order_by_stream[str(stream)] = {
            str(row.event_id): index for index, row in enumerate(
                part.sort_values(["_ts", "_athena_input_order", "event_id"], kind="stable")
                .itertuples(index=False)
            )
        }
    shifted = replay_df.copy()
    shifted["timestamp"] = shifted["timestamp"].map(_raw_timestamp)
    inserted_frames = []
    inserted_provenance = manifest.get("inserted_benign_events")
    if not isinstance(inserted_provenance, list) or not inserted_provenance:
        raise ValueError("event-level replay manifest contains no source-event provenance")
    for gap_index, gap in enumerate(gaps):
        before = gap.get("before") or {}
        after = gap.get("after") or {}
        stream = str(gap.get("host") or "")
        event_order = event_order_by_stream.get(stream, {})
        before_id, after_id = str(before.get("event_id") or ""), str(after.get("event_id") or "")
        if before_id not in event_order or after_id not in event_order or event_order[before_id] >= event_order[after_id]:
            raise ValueError("attack-event boundaries do not join an ordered held-out E3 event pair")
        for ref, expected_scene in (
            (before, str(gap.get("source_scene") or "")),
            (after, str(gap.get("next_source_scene") or "")),
        ):
            sid, edge_index = int(ref["snapshot"]), int(ref["edge_index"])
            graph = snapshots[sid]
            if (
                sid not in test_id_set
                or str(graph["source_scene"] or "") != expected_scene
                or str(graph["host_id"] or "") != stream
                or event_sha256(graph, edge_index) != ref.get("event_sha256")
            ):
                raise ValueError("attack-event boundary no longer matches the held-out audit event")
        seconds = float(gap["interval_seconds"])
        raw_delta = int(seconds * 1_000_000_000)
        stream_mask = shifted["host_id"].astype(str).eq(stream)
        later_mask = shifted["event_id"].astype(str).map(event_order).gt(event_order[before_id]).fillna(False)
        shifted.loc[stream_mask & later_mask, "timestamp"] += raw_delta

        source_ids = [int(value) for value in gap.get("source_snapshots", [])]
        if not source_ids or any(value not in train_ids for value in source_ids):
            raise ValueError("replay gap source slice is not wholly training-only")
        for source_id, expected_hash in zip(source_ids, gap.get("source_snapshot_sha256", [])):
            source = snapshots[source_id]
            if (
                graph_sha256(source) != expected_hash
                or any(int(value or 0) for value in source.vs["label"])
                or str(source["host_id"] or "") != stream
            ):
                raise ValueError("replay gap source slice is changed or not benign-only")
        source_event_ids = {
            event_id for source_id in source_ids for event_id in _event_ids(snapshots[source_id], source_id)
        }
        source_df = _event_dataframe(handler, source_event_ids)
        raw_rows = {
            str(row["event_id"]): row for _, row in source_df.iterrows()
        }
        expected_rows = []
        for source_id in source_ids:
            source_graph = snapshots[source_id]
            for edge_index, edge in enumerate(source_graph.es):
                event_id = str(edge.attributes().get("event_id") or f"snapshot-{source_id}-edge-{edge_index}")
                payload = canonical_event_row_payload(raw_rows[event_id])
                expected_rows.append((source_id, event_id, payload, f"snapshot:{source_id}/edge:{edge_index}"))
        bound_rows = [row for row in inserted_provenance
                      if str(row.get("attack_event_id") or "") == str(gap["attack_event_id"])]
        if len(bound_rows) != len(expected_rows):
            raise ValueError("source plan does not bind every inserted benign audit event")
        for source_id, event_id, payload, locator in expected_rows:
            digest = canonical_event_row_sha256(raw_rows[event_id])
            matches = [row for row in bound_rows
                       if int(row.get("source_snapshot", -1)) == source_id
                       and row.get("source_event_id") == event_id]
            if len(matches) != 1 or matches[0].get("source_event_payload") != payload \
                    or matches[0].get("source_event_hash") != digest or matches[0].get("source_locator") != locator:
                raise ValueError("inserted benign event provenance differs from the source audit edge")
        source_snapshot_by_event = {
            event_id: source_id
            for source_id in source_ids
            for event_id in _event_ids(snapshots[source_id], source_id)
        }
        source_df["timestamp"] = source_df["timestamp"].map(_raw_timestamp)
        source_start = int(gap["source_start_timestamp"])
        source_end = int(gap["source_end_timestamp"])
        validate_source_event_window(source_df, source_start, source_end, stream)
        before_raw = _raw_timestamp(replay_df.loc[
            replay_df["event_id"].astype(str).eq(before_id) &
            replay_df["host_id"].astype(str).eq(stream), "timestamp"
        ].iloc[0])
        before_shift = before_raw + raw_delta * sum(
            str(prior.get("host") or "") == stream
            and event_order[str((prior.get("before") or {}).get("event_id") or "")] <= event_order[before_id]
            for prior in gaps[:gap_index]
        )
        source_df["timestamp"] = before_shift + 1 + (source_df["timestamp"] - source_start)
        source_df["source_event_id"] = source_df["event_id"].astype(str)
        source_df["replay_source_snapshot"] = source_df["source_event_id"].map(source_snapshot_by_event)
        source_df["replay_attack_id"] = str(gap["attack_event_id"])
        source_df["replay_condition"] = str(manifest["condition"])
        source_df["event_id"] = [
            f"replay:{manifest['condition']}:{gap['attack_event_id']}:{gap_index}:{index}:{value}"
            for index, value in enumerate(source_df["source_event_id"])
        ]
        inserted_frames.append(source_df)

    if not set(shifted["event_id"].astype(str)) == original_ids:
        raise ValueError("event-level replay changed held-out event identities")
    combined = pd.concat([shifted, *inserted_frames], ignore_index=True, sort=False)
    combined.sort_values(["timestamp", "_athena_input_order", "event_id"], kind="stable", inplace=True)
    inserted_ids = _replace_test_snapshots(handler, test_ids, combined)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return inserted_ids, {
        "path": str(manifest_path), "sha256": digest, "condition": manifest["condition"],
        "attack_event_boundaries_sha256": boundary["sha256"],
        "replay_gap_count": len(gaps), "replayed_test_snapshots": len(test_ids) + len(inserted_ids),
    }


__all__ = ["apply_interval_replay", "event_sha256", "graph_sha256"]
