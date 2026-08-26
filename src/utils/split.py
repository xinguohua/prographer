"""Shared train/test split contract for ATHENA artifact entry points.

The paper protocol assigns every benign-only day to training, splits attack
days chronologically, and holds the later attack days out for evaluation.
Augmentation and detection must call this same implementation before they
select benign anchors or attack donors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math


SPLIT_MODE = "date_partition_benign_days_and_attack_days"
ATLAS_SPLIT_MODE = "atlas_original_leave_one_attack_out"


def _chronological_split(items, train_ratio: float):
    items = list(items)
    if not items:
        return [], []
    if len(items) == 1:
        return items, []
    cut = int(round(len(items) * float(train_ratio)))
    cut = max(1, min(cut, len(items) - 1))
    return items[:cut], items[cut:]


def _snapshot_time_key(handler, snapshot_id: int) -> tuple:
    snapshots = getattr(handler, "snapshots", [])
    if snapshot_id < 0 or snapshot_id >= len(snapshots):
        return (1, snapshot_id)
    graph = snapshots[snapshot_id]
    values = []
    try:
        if graph is not None and graph.vcount() > 0 and "timestamp" in graph.vs.attributes():
            values.extend(float(value) for value in graph.vs["timestamp"] if value is not None)
    except Exception:
        pass
    try:
        if graph is not None and graph.ecount() > 0 and "timestamp" in graph.es.attributes():
            values.extend(float(value) for value in graph.es["timestamp"] if value is not None)
    except Exception:
        pass
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return (1, snapshot_id)
    return (0, min(values), snapshot_id)


def _snapshot_day_key(handler, snapshot_id: int):
    time_key = _snapshot_time_key(handler, snapshot_id)
    if time_key[0] != 0:
        return ("unknown", snapshot_id)
    timestamp = float(time_key[1])
    if timestamp > 1e18:
        timestamp /= 1e9
    elif timestamp > 1e15:
        timestamp /= 1e6
    elif timestamp > 1e12:
        timestamp /= 1e3
    try:
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).date().isoformat()
    except Exception:
        return ("unknown", snapshot_id)


def _snapshot_has_attack(handler, snapshot_id: int) -> bool:
    snapshots = getattr(handler, "snapshots", [])
    if snapshot_id < 0 or snapshot_id >= len(snapshots):
        return False
    graph = snapshots[snapshot_id]
    try:
        if graph is not None and graph.vcount() > 0 and "label" in graph.vs.attributes():
            return any(int(value or 0) == 1 for value in graph.vs["label"])
    except Exception:
        return False
    return False


def build_split(handler, train_ratio: float):
    """Return deterministic train/test snapshot IDs and auditable metadata."""
    atlas_fold = str(getattr(handler, "atlas_fold", "") or "").upper()
    atlas_scenarios = tuple(getattr(handler, "atlas_scenarios", ()) or ())
    if atlas_fold and atlas_scenarios:
        if atlas_fold not in atlas_scenarios:
            raise RuntimeError(f"ATLAS fold {atlas_fold} is outside {atlas_scenarios}")
        snapshots = getattr(handler, "snapshots", [])
        scenario_ids = list(getattr(handler, "snapshot_scenarios", []) or [])
        if len(scenario_ids) != len(snapshots):
            scenario_ids = [
                str(graph["atlas_scenario"])
                if graph is not None and "atlas_scenario" in graph.attributes()
                else ""
                for graph in snapshots
            ]
        train_scenarios = [scenario for scenario in atlas_scenarios if scenario != atlas_fold]
        train_ids = [
            snapshot_id for snapshot_id, scenario in enumerate(scenario_ids)
            if scenario in train_scenarios
        ]
        test_ids = [
            snapshot_id for snapshot_id, scenario in enumerate(scenario_ids)
            if scenario == atlas_fold
        ]
        if not train_ids or not test_ids:
            raise RuntimeError(
                f"ATLAS fold {atlas_fold} requires non-empty train and test scenarios"
            )
        train_ids = sorted(train_ids, key=lambda sid: (scenario_ids[sid], _snapshot_time_key(handler, sid)))
        test_ids = sorted(test_ids, key=lambda sid: _snapshot_time_key(handler, sid))
        return train_ids, test_ids, {
            "mode": ATLAS_SPLIT_MODE,
            "fold": atlas_fold,
            "family": atlas_fold[0],
            "train_ratio": None,
            "train_scenarios": train_scenarios,
            "test_scenarios": [atlas_fold],
            "train_snapshots": train_ids,
            "test_snapshots": test_ids,
            "ordered_snapshots": train_ids + test_ids,
        }

    all_ids = list(range(len(getattr(handler, "snapshots", []))))
    ordered_ids = sorted(all_ids, key=lambda sid: _snapshot_time_key(handler, sid))
    day_to_ids: dict[object, list[int]] = {}
    for snapshot_id in ordered_ids:
        day = _snapshot_day_key(handler, snapshot_id)
        if isinstance(day, tuple) and day and day[0] == "unknown":
            raise RuntimeError(
                f"snapshot {snapshot_id} has no valid timestamp; date split cannot be audited"
            )
        day_to_ids.setdefault(day, []).append(snapshot_id)

    benign_days = []
    attack_days = []
    for day, snapshot_ids in day_to_ids.items():
        if any(_snapshot_has_attack(handler, snapshot_id) for snapshot_id in snapshot_ids):
            attack_days.append(day)
        else:
            benign_days.append(day)

    benign_days = sorted(benign_days, key=str)
    attack_days = sorted(attack_days, key=str)
    train_ids = [snapshot_id for day in benign_days for snapshot_id in day_to_ids[day]]

    attack_train_days, attack_test_days = _chronological_split(attack_days, train_ratio)
    if attack_test_days:
        train_ids.extend(
            snapshot_id
            for day in attack_train_days
            for snapshot_id in day_to_ids[day]
        )
        test_ids = [
            snapshot_id
            for day in attack_test_days
            for snapshot_id in day_to_ids[day]
        ]
    elif attack_train_days:
        train_ids.extend(
            snapshot_id
            for day in attack_train_days
            for snapshot_id in day_to_ids[day]
        )
        test_ids = []
    else:
        test_ids = []

    train_ids = sorted(set(train_ids), key=lambda sid: _snapshot_time_key(handler, sid))
    test_ids = sorted(set(test_ids), key=lambda sid: _snapshot_time_key(handler, sid))
    return train_ids, test_ids, {
        "mode": SPLIT_MODE,
        "train_ratio": float(train_ratio),
        "benign_train_days": benign_days,
        "attack_train_days": attack_train_days if attack_days else [],
        "attack_test_days": attack_test_days if attack_days else [],
        "train_snapshots": train_ids,
        "test_snapshots": test_ids,
        "ordered_snapshots": ordered_ids,
    }
