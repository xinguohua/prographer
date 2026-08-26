"""Evaluate Proof Tables VII/VIII from source-linked GT and replay outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interpretation.global_alignment import (
    load_technique_sequence_records,
)

TECHNIQUE_RE = re.compile(r"T\d{4}(?:\.\d{3})?")
SEQUENCE_CONDITIONS = ("Basic", "24h", "48h", "72h", "E5")
E5_DATASETS = {"cadets5", "theia5", "trace5", "clearscope5"}
E3_DATASETS = {"cadets", "theia", "trace", "clearscope"}
MAPPING_VARIANTS = ("direct", "tech-enhanced", "log-enhanced", "full-enhanced")
E5_SOURCE_DATASET = {
    "cadets5": "cadets", "theia5": "theia",
    "trace5": "trace", "clearscope5": "clearscope",
}


def _complete_checkpoint(checkpoint: dict) -> bool:
    return bool(
        checkpoint.get("source_run_mode") == "complete"
        and checkpoint.get("source_variant") == "full-athena"
        and (checkpoint.get("source_augmentation") or {}).get("manifest_sha256")
    )


def _provenance_ok(row: dict) -> bool:
    return bool(
        row.get("source_id") and row.get("source_record") and row.get("source_corpus")
        and re.fullmatch(r"[0-9a-fA-F]{64}", str(row.get("source_hash", "")))
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _validate_portable_mapping_evidence(registry_path: Path, row: dict, boundary: dict) -> None:
    source_record = Path(str(row.get("source_record") or ""))
    if not source_record.is_absolute():
        source_record = registry_path.parent / source_record
    if not source_record.is_file() or _sha256(source_record) != str(row.get("source_hash") or "").lower():
        raise ValueError("portable mapping source-record file/hash mismatch")
    record_hash = str(row.get("source_record_sha256") or "").lower()
    evidence = None
    for line in source_record.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        if str(candidate.get("record_sha256") or "").lower() == record_hash:
            evidence = candidate
            break
    if not isinstance(evidence, dict):
        raise ValueError("portable mapping source evidence is absent")
    payload = dict(evidence)
    claimed = str(payload.pop("record_sha256", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", claimed) or _canonical_hash(payload) != claimed:
        raise ValueError("portable mapping source evidence hash does not recompute")
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
        raise ValueError("portable mapping boundary contradicts its source evidence")


def _validate_atlas_mapping_evidence(registry_path: Path, row: dict, boundary: dict) -> None:
    """Rejoin a final ATLAS mapping to its reviewed official-v1 endpoint row."""
    from scripts.import_atlas_v1_attack_annotations import BOUNDARY_PINS, _boundary_pin
    source_record = Path(str(row.get("source_record") or ""))
    if not source_record.is_absolute():
        source_record = registry_path.parent / source_record
    if not source_record.is_file() or _sha256(source_record) != str(row.get("source_hash") or "").lower():
        raise ValueError("ATLAS mapping source-record file/hash mismatch")
    record_hash = str(row.get("source_record_sha256") or "").lower()
    evidence = None
    for line in source_record.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        if str(candidate.get("record_sha256") or "").lower() == record_hash:
            evidence = candidate
            break
    if not isinstance(evidence, dict):
        raise ValueError("ATLAS mapping source evidence is absent")
    payload = dict(evidence)
    claimed = str(payload.pop("record_sha256", "")).lower()
    if _canonical_hash(payload) != claimed:
        raise ValueError("ATLAS mapping source evidence hash does not recompute")
    if evidence.get("schema") != "athena.atlas_v1.attack_annotation_evidence.v1":
        raise ValueError("ATLAS mapping uses an unsupported evidence schema")
    endpoint = evidence.get("endpoint_record") if isinstance(evidence.get("endpoint_record"), dict) else {}
    endpoint_boundary = evidence.get("selected_boundary") if isinstance(evidence.get("selected_boundary"), dict) else {}
    endpoint_registry = evidence.get("endpoint_registry") if isinstance(evidence.get("endpoint_registry"), dict) else {}
    source_registry = source_record.parent / str(endpoint_registry.get("path") or "")
    if not source_registry.is_file() or _sha256(source_registry) != str(endpoint_registry.get("sha256") or ""):
        raise ValueError("ATLAS endpoint registry file/hash mismatch")
    official_endpoint = None
    for line in source_registry.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = json.loads(line)
        if candidate.get("source_id") == endpoint_registry.get("source_id"):
            official_endpoint = candidate
            break
    if official_endpoint != endpoint or endpoint.get("derived_sha256") != endpoint_registry.get("derived_sha256"):
        raise ValueError("ATLAS mapping evidence contradicts the official endpoint registry")
    endpoint_source_id = str(endpoint_registry.get("source_id") or "")
    if endpoint_source_id not in BOUNDARY_PINS or _boundary_pin(endpoint_boundary) != BOUNDARY_PINS[endpoint_source_id]:
        raise ValueError("ATLAS mapping selected event differs from the reviewed boundary pin")
    expected_boundary = {
        "snapshot": endpoint_boundary.get("snapshot"),
        "source_global_snapshot": endpoint_boundary.get("source_global_snapshot"),
        "event_id": endpoint_boundary.get("event_id"),
        "anchor": endpoint_boundary.get("anchor"),
        "event_sha256": endpoint_boundary.get("event_sha256"),
        "source_event_sha256": endpoint_boundary.get("event_sha256"),
        "action": endpoint_boundary.get("action"),
        "timestamp": endpoint_boundary.get("event_timestamp"),
    }
    if any(str(boundary.get(key) or "") != str(value or "") for key, value in expected_boundary.items()):
        raise ValueError("ATLAS mapping boundary contradicts its exact official-v1 event")
    taxonomy = evidence.get("attack_taxonomy") if isinstance(evidence.get("attack_taxonomy"), dict) else {}
    if (
        row.get("scene") != endpoint.get("scene")
        or row.get("host_id") != endpoint.get("host")
        or row.get("anchor_uuid") != endpoint_boundary.get("anchor")
        or row.get("reference_technique") != taxonomy.get("technique")
        or row.get("reference_tactic") != taxonomy.get("tactic")
    ):
        raise ValueError("ATLAS mapping fields contradict the reviewed endpoint evidence")


def load_replay_runs(paths: list[Path]) -> list[dict]:
    """Load 24/48/72h outputs produced by a complete detector+interpreter rerun."""
    runs = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not _provenance_ok(manifest):
            raise ValueError(f"replay manifest {path} lacks source-linked provenance")
        source_record = Path(str(manifest["source_record"]))
        if not source_record.is_absolute():
            source_record = path.parent / source_record
        if not source_record.is_file() or _sha256(source_record) != str(manifest["source_hash"]).lower():
            raise ValueError(f"replay manifest {path} source spec hash does not match")
        condition = str(manifest.get("condition") or "")
        dataset = str(manifest.get("dataset") or "")
        if condition not in {"24h", "48h", "72h"} or dataset not in E3_DATASETS:
            raise ValueError(f"replay manifest {path} is not an E3 interval rerun")
        artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
        loaded = {}
        for name in ("detection", "interpretation", "source_event_manifest", "checkpoint"):
            item = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
            artifact_path = Path(str(item.get("path") or ""))
            if not artifact_path.is_absolute():
                artifact_path = path.parent / artifact_path
            digest = str(item.get("sha256") or "").lower()
            if not artifact_path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"replay manifest {path} lacks {name} artifact")
            if _sha256(artifact_path) != digest:
                raise ValueError(f"replay manifest {path} has a changed {name} artifact")
            if name != "checkpoint":
                loaded[name] = json.loads(artifact_path.read_text(encoding="utf-8"))
        detection, interp, events = (
            loaded["detection"], loaded["interpretation"], loaded["source_event_manifest"]
        )
        replay_sha = str(artifacts["source_event_manifest"]["sha256"]).lower()
        checkpoint_sha = str(artifacts["checkpoint"]["sha256"]).lower()
        if (
            (detection.get("interval_replay") or {}).get("sha256") != replay_sha
            or (interp.get("interval_replay") or {}).get("sha256") != replay_sha
            or (detection.get("checkpoint") or {}).get("sha256") != checkpoint_sha
            or (interp.get("detection_checkpoint") or {}).get("sha256") != checkpoint_sha
        ):
            raise ValueError(f"replay manifest {path} outputs did not consume the bound injection")
        scene = str(manifest.get("scene") or "")
        if any(
            str(payload.get("dataset") or "") != dataset
            or str(payload.get("scene") or "") != scene
            for payload in (detection, interp)
        ):
            raise ValueError(f"replay manifest {path} dataset/scene does not match rerun outputs")
        inserted = events.get("inserted_benign_events") if isinstance(events, dict) else None
        if not isinstance(inserted, list) or not inserted or any(
            not isinstance(row, dict)
            or int(row.get("label", -1)) != 0
            or not row.get("source_event_id")
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_event_hash", "")).lower())
            for row in inserted
        ):
            raise ValueError(f"replay manifest {path} lacks source-hashed benign E3 events")
        for row in inserted:
            payload = row.get("source_event_payload")
            expected = (
                hashlib.sha256(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                if isinstance(payload, dict) else ""
            )
            if expected != str(row["source_event_hash"]).lower() or not row.get("source_locator"):
                raise ValueError(f"replay manifest {path} event provenance does not recompute")
        if interp.get("mapping_variant") != "full-enhanced":
            raise ValueError(f"replay manifest {path} must use full-enhanced interpretation")
        runs.append({**manifest, "_interpretation": interp})
    return runs


def load_ground_truth(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not _provenance_ok(row):
            raise ValueError(f"GT line {number} lacks source-linked provenance")
        record_type = str(row.get("record_type", "mapping"))
        if record_type == "mapping":
            anchor = row.get("anchor_uuid") or row.get("uuid")
            row = {**row, "anchor_uuid": str(anchor or "")}
            required = ("dataset", "scene", "host_id", "anchor_uuid", "reference_technique", "reference_tactic")
            if any(row.get(key) in (None, "") for key in required):
                raise ValueError(f"mapping GT line {number} lacks entity technique/tactic fields")
            boundary = row.get("boundary") or {}
            portable_keys = (
                "event_id", "anchor", "actor", "object", "action", "timestamp",
                "source_event_sha256",
            )
            portable = boundary.get("snapshot") in (None, "")
            if row.get("scene") == "*" and row.get("scene_resolution") == "runtime_unique_raw_event":
                portable_keys = (*portable_keys, "raw_event_uuid")
            if (
                any(boundary.get(key) in (None, "") for key in portable_keys)
                if portable else
                any(boundary.get(key) in (None, "") for key in ("snapshot", "event_id", "anchor", "event_sha256"))
            ):
                raise ValueError(f"mapping GT line {number} lacks an exact source-linked event boundary")
            if row.get("dataset") == "atlas":
                _validate_atlas_mapping_evidence(path, row, boundary)
            if portable:
                _validate_portable_mapping_evidence(path, row, boundary)
            technique = str(row["reference_technique"]).upper()
            if not TECHNIQUE_RE.fullmatch(technique):
                raise ValueError(f"mapping GT line {number} has invalid ATT&CK technique")
            row = {**row, "reference_technique": technique}
        elif record_type == "sequence":
            prediction_id = row.get("prediction_id") or row.get("attack_id")
            boundary = row.get("boundary") or {}
            row = {**row, "attack_id": str(prediction_id or "")}
            required = ("dataset", "scene", "host_id", "attack_id")
            if (
                any(row.get(key) in (None, "") for key in required)
                or not isinstance(row.get("reference_tactic_sequence"), list)
                or not row.get("reference_tactic_sequence")
                or any(boundary.get(key) in (None, "") for key in ("snapshot", "event_id", "anchor", "event_sha256"))
            ):
                raise ValueError(f"sequence GT line {number} lacks attack reference fields")
            if "candidate_tactic_chains" in row or "condition" in row:
                raise ValueError(
                    f"sequence GT line {number} contains prediction fields; "
                    "predictions must come from interpretation/replay output"
                )
        elif record_type == "attack_event":
            boundary = row.get("boundary") or {}
            if (
                any(row.get(key) in (None, "") for key in ("dataset", "scene"))
                or not (row.get("host_id") or row.get("host"))
                or any(boundary.get(key) in (None, "") for key in ("snapshot", "event_id", "anchor", "event_sha256"))
            ):
                raise ValueError(f"attack_event GT line {number} lacks an exact source-linked boundary")
        else:
            raise ValueError(f"GT line {number} has unknown record_type")
        rows.append({**row, "record_type": record_type})
    if not rows:
        raise ValueError("ground-truth JSONL is empty")
    return rows


def _mapping_index(interps: list[dict]) -> tuple[dict, list[dict]]:
    index = {}
    for interp in interps:
        dataset = str(interp.get("dataset") or "")
        variant = str(interp.get("mapping_variant") or "")
        if variant not in MAPPING_VARIANTS:
            raise ValueError("interpretation output lacks a valid Table VII mapping_variant")
        for row in interp.get("final_mapping_predictions", []):
            scene = str(row.get("source_scene") or "")
            anchor = str(row.get("anchor_uuid") or row.get("uuid") or "")
            host = str(row.get("host_id") or "")
            snapshot = row.get("boundary_snapshot")
            event_id = str(row.get("boundary_event_id") or "")
            if not all((dataset, scene, host, anchor, event_id)) or snapshot in (None, ""):
                raise ValueError(
                    "interpretation mapping row lacks dataset/source_scene/host/anchor/boundary"
                )
            candidates = list(row.get("top_k_candidates", []))
            reason = row.get("unmapped_reason")
            if candidates and reason:
                raise ValueError("mapped interpretation row must not carry unmapped_reason")
            if not candidates:
                if reason == "detector_miss":
                    raise ValueError("Table VII registry anchor was not emitted by the detector")
                if reason not in {"no_causal_path_query", "top_similarity_below_gamma"}:
                    raise ValueError("empty mapping candidates require a controlled unmapped_reason")
                if reason == "top_similarity_below_gamma" and not (
                    float(row.get("max_similarity", float("inf"))) < float(row.get("gamma", float("-inf")))
                ):
                    raise ValueError("Unm requires max_similarity < gamma")
            key = (variant, dataset, scene, host, anchor, event_id)
            if key in index:
                raise ValueError(f"duplicate interpretation mapping key: {key}")
            index[key] = {
                "candidates": candidates,
                "source_id": str(row.get("source_id") or ""),
                "snapshot": int(snapshot),
                "event_sha256": str(row.get("boundary_event_sha256") or ""),
            }
    return index, []


def _prediction_key_for_truth(index: dict, variant: str, gt: dict) -> tuple:
    """Resolve one exact prediction, including portable E5 runtime scenes."""
    dataset = str(gt["dataset"])
    scene = str(gt["scene"])
    host = str(gt["host_id"])
    anchor = str(gt.get("anchor_uuid") or gt.get("uuid") or "")
    event_id = str((gt.get("boundary") or {})["event_id"])
    if scene != "*":
        key = (variant, dataset, scene, host, anchor, event_id)
        if key not in index:
            raise ValueError(
                f"mapping variant {variant} lacks evaluated detector-positive key {key[1:]}"
            )
        return key
    if gt.get("scene_resolution") != "runtime_unique_raw_event":
        raise ValueError("wildcard GT scene lacks the runtime-unique resolution contract")
    matches = [
        key for key in index
        if key[0] == variant
        and key[1] == dataset
        and key[3] == host
        and key[4] == anchor
        and key[5] == event_id
    ]
    if len(matches) != 1:
        raise ValueError(
            "portable wildcard GT must join exactly one runtime scene: "
            f"variant={variant} dataset={dataset} host={host} anchor={anchor} "
            f"event={event_id} matches={len(matches)}"
        )
    return matches[0]


def score_mapping(interps: list[dict], truth: list[dict], top_ks=(1, 3, 5)) -> dict:
    """Table VII: Acc / STE / CTE / Unmapped for parent-technique mapping."""
    index, join_audit = _mapping_index(interps)
    output = {
        variant: {
            str(k): {name: 0 for name in ("Acc", "STE", "CTE", "Unm")} for k in top_ks
        } for variant in MAPPING_VARIANTS
    }
    mapping_truth = [gt for gt in truth if gt.get("record_type") == "mapping"]
    for variant in MAPPING_VARIANTS:
        matched_keys = {_prediction_key_for_truth(index, variant, gt) for gt in mapping_truth}
        artifact_keys = {key for key in index if key[0] == variant}
        if matched_keys != artifact_keys or len(matched_keys) != len(mapping_truth):
            raise ValueError(
                f"mapping variant {variant} coverage differs from GT: "
                f"matched={len(matched_keys)} truth={len(mapping_truth)} "
                f"extra={sorted(artifact_keys - matched_keys)}"
            )
    evaluated = 0
    for gt in truth:
        if gt["record_type"] != "mapping":
            continue
        evaluated += 1
        for variant in MAPPING_VARIANTS:
            artifact_key = _prediction_key_for_truth(index, variant, gt)
            prediction = index[artifact_key]
            boundary = gt.get("boundary") or {}
            if boundary.get("snapshot") not in (None, ""):
                if (
                    prediction["snapshot"] != int(boundary["snapshot"])
                    or prediction["event_sha256"] != str(boundary.get("event_sha256") or "")
                ):
                    raise ValueError("mapping prediction resolved a different snapshot/event hash than GT")
            elif (
                prediction["source_id"] != str(gt.get("source_id") or "")
                or not re.fullmatch(r"[0-9a-f]{64}", prediction["event_sha256"])
            ):
                raise ValueError("portable mapping prediction lacks its source ID or resolved event hash")
            candidates = prediction["candidates"]
            for k in top_ks:
                rows = candidates[:k]
                techniques = {
                    str(row.get("technique", "")).upper().split(".", 1)[0] for row in rows
                }
                tactics = {
                    str(tactic) for row in rows
                    for tactic in (row.get("tactics") or [row.get("tactic")]) if tactic
                }
                if not rows:
                    label = "Unm"
                elif gt["reference_technique"].split(".", 1)[0] in techniques:
                    label = "Acc"
                elif str(gt["reference_tactic"]) in tactics:
                    label = "STE"
                else:
                    label = "CTE"
                output[variant][str(k)][label] += 1
    return {"records_per_variant": evaluated, "variants": output, "join_audit": join_audit}


def _lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for value in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1 if value == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _sequence_key(row: dict) -> tuple[str, str, str, str]:
    return tuple(str(row.get(key) or "") for key in ("dataset", "scene", "host_id", "attack_id"))


def score_sequence_conditions(
    predictions: list[dict],
    truth: list[dict],
    attackseq_records: list[dict],
    top_ks=(1, 3, 5),
) -> dict:
    """Table VIII: Basic/24h/48h/72h/E5 FM/PM/Miss via LCS/min."""
    counts = {
        condition: {
            str(k): {"FM": 0, "PM": 0, "Miss": 0} for k in top_ks
        } for condition in SEQUENCE_CONDITIONS
    }
    known_sources = {str(record.get("source_id")) for record in attackseq_records}
    truth_index = {
        _sequence_key(row): row for row in truth if row.get("record_type") == "sequence"
    }
    if len(truth_index) != sum(row.get("record_type") == "sequence" for row in truth):
        raise ValueError("duplicate sequence ground-truth key")
    prediction_index = {}
    for row in predictions:
        condition = str(row.get("condition") or "")
        key = _sequence_key(row)
        if condition not in SEQUENCE_CONDITIONS or not all(key):
            raise ValueError("sequence prediction lacks condition/dataset/scene/host/attack_id")
        if condition == "E5" and key[0] not in E5_DATASETS:
            raise ValueError("Unseen/E5 condition requires a DARPA E5 dataset key")
        composite = (condition, key)
        if composite in prediction_index:
            raise ValueError(f"duplicate sequence prediction: {composite}")
        prediction_index[composite] = row

    expected = set()
    for key in truth_index:
        if key[0] in E5_DATASETS:
            expected.add(("E5", key))
        elif key[0] in E3_DATASETS:
            expected.update((condition, key) for condition in ("Basic", "24h", "48h", "72h"))
        else:
            expected.add(("Basic", key))
    actual = set(prediction_index)
    if actual != expected:
        raise ValueError(
            "sequence prediction coverage differs from GT: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )

    details = []
    for (condition, key), row in prediction_index.items():
        gt = truth_index.get(key)
        if gt is None:
            raise ValueError(f"sequence prediction has no matching ground truth: {condition}/{key}")
        candidates = []
        for value in row.get("candidate_tactic_chains", []):
            if isinstance(value, dict):
                chain = list(value.get("tactics", []))
                source_id = str(
                    value.get("attackseq_source_id")
                    or (value.get("library_source") or {}).get("source_id") or ""
                )
                if source_id and source_id not in known_sources:
                    raise ValueError(f"unknown AttackSeqBench candidate source: {source_id}")
            else:
                chain, source_id = list(value), ""
            if chain:
                candidates.append((chain, source_id))
        reference = list(gt["reference_tactic_sequence"])
        row_details = {}
        for k in top_ks:
            ranked = candidates[:k]
            best_lcs, best_source = 0, ""
            for chain, source_id in ranked:
                value = _lcs_length(chain, reference)
                if value > best_lcs:
                    best_lcs, best_source = value, source_id
            label = "FM" if best_lcs == len(reference) else ("PM" if best_lcs > 0 else "Miss")
            counts[condition][str(k)][label] += 1
            row_details[str(k)] = {
                "classification": label,
                "lcs_length": best_lcs,
                "reference_length": len(reference),
                "attackseq_source_id": best_source or None,
            }
        details.append({
            "source_id": gt["source_id"], "condition": condition,
            "dataset": key[0], "scene": key[1], "host_id": key[2], "attack_id": key[3],
            "top_k": row_details,
        })
    return {"condition_counts": counts, "per_record": details}


def sequence_predictions_from_outputs(
    interps: list[dict],
    replay_runs: list[dict],
    attackseq_records: list[dict],
) -> list[dict]:
    """Join only actual interpretation/replay outputs to Table VIII keys."""
    predictions = []
    basic_checkpoint_hashes = {}
    for interp in interps:
        if interp.get("mapping_variant") != "full-enhanced":
            continue
        dataset = str(interp.get("dataset") or "")
        alignments = interp.get("final_attack_predictions") or []
        if not alignments:
            continue
        checkpoint = interp.get("detection_checkpoint") or {}
        if dataset in E5_DATASETS and (
            checkpoint.get("source_dataset") != E5_SOURCE_DATASET[dataset]
            or not _complete_checkpoint(checkpoint)
        ):
            raise ValueError("E5/Unseen interpretation requires its paired complete E3 checkpoint")
        root_scene = str(interp.get("scene") or "")
        if dataset in E3_DATASETS:
            checkpoint_hash = str(checkpoint.get("sha256") or "")
            if not re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash):
                raise ValueError("Basic E3 interpretation lacks the frozen train-save checkpoint hash")
            if not _complete_checkpoint(checkpoint):
                raise ValueError("Basic E3 interpretation checkpoint is not complete ATHENA")
            basic_checkpoint_hashes[(dataset, root_scene)] = checkpoint_hash
        for alignment in alignments:
            host_id = str(alignment.get("host_id") or "")
            scene = str(alignment.get("source_scene") or root_scene or "")
            attack_id = str(alignment.get("attack_id") or "")
            chains = alignment.get("aligned_top_k_chains")
            if not all((dataset, scene, str(host_id), attack_id)) or not isinstance(chains, list):
                raise ValueError("final_attack_prediction lacks exact sequence join fields")
            predictions.append({
                "condition": "E5" if dataset in E5_DATASETS else "Basic",
                "dataset": dataset,
                "scene": scene,
                "host_id": str(host_id),
                "attack_id": attack_id,
                "candidate_tactic_chains": chains,
            })

    for run in replay_runs:
        dataset = str(run["dataset"])
        scene = str(run.get("scene") or "")
        condition = str(run["condition"])
        interp = run["_interpretation"]
        replay_checkpoint = str((interp.get("detection_checkpoint") or {}).get("sha256") or "")
        if replay_checkpoint != basic_checkpoint_hashes.get((dataset, scene)):
            raise ValueError("interval replay did not reuse the Basic E3 checkpoint")
        for alignment in (interp.get("final_attack_predictions") or []):
            host_id = str(alignment.get("host_id") or "")
            attack_id = str(alignment.get("attack_id") or "")
            source_scene = str(alignment.get("source_scene") or scene or "")
            chains = alignment.get("aligned_top_k_chains")
            if not attack_id or not source_scene or not isinstance(chains, list):
                raise ValueError("replay interpretation lacks attack_id/aligned chains")
            predictions.append({
                "condition": condition,
                "dataset": dataset,
                "scene": source_scene,
                "host_id": str(host_id),
                "attack_id": attack_id,
                "candidate_tactic_chains": chains,
            })
    return predictions


def score(
    interps: dict | list[dict],
    truth: list[dict],
    attackseq_records: list[dict] | None = None,
    replay_runs: list[dict] | None = None,
) -> dict:
    interp_list = [interps] if isinstance(interps, dict) else list(interps)
    library = attackseq_records or []
    has_sequence_truth = any(row.get("record_type") == "sequence" for row in truth)
    predictions = (
        sequence_predictions_from_outputs(interp_list, replay_runs or [], library)
        if has_sequence_truth else []
    )
    return {
        "table_vii_mapping": score_mapping(interp_list, truth),
        "table_viii_sequence": score_sequence_conditions(predictions, truth, library),
    }


def validate_registry_bindings(interps: list[dict], replay_runs: list[dict], truth_hash: str) -> None:
    """Bind every Table VII branch and Table VIII replay to one scored registry."""
    for interp in interps:
        if (interp.get("attack_event_boundaries") or {}).get("sha256") != truth_hash:
            raise ValueError("interpretation predictions are not bound to the scored GT registry hash")
    for run in replay_runs:
        if (run.get("artifacts", {}).get("attack_event_boundaries") or {}).get("sha256") != truth_hash:
            raise ValueError("replay predictions are not bound to the scored GT registry hash")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpretation", required=True, action="append", type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--attackseq-records", required=True, type=Path)
    parser.add_argument(
        "--replay-manifest", action="append", type=Path, default=[],
        help="manifest from a complete 24/48/72h detection+interpretation rerun",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-predictions", type=int, default=53)
    args = parser.parse_args(argv)
    interps = []
    for path in args.interpretation:
        interp = json.loads(path.read_text(encoding="utf-8"))
        interps.append({**interp, "_artifact_sha256": _sha256(path)})
    replay_runs = load_replay_runs(args.replay_manifest)
    truth = load_ground_truth(args.ground_truth)
    expected = int(args.expected_predictions)
    mapping_count = sum(row["record_type"] == "mapping" for row in truth)
    e3_count = sum(row["record_type"] == "sequence" and row["dataset"] in E3_DATASETS for row in truth)
    e5_count = sum(row["record_type"] == "sequence" and row["dataset"] in E5_DATASETS for row in truth)
    if (mapping_count, e3_count, e5_count) != (expected, expected, expected):
        raise ValueError(
            f"paper-profile GT requires exactly {expected} mapping, E3 sequence, and E5 sequence rows"
        )
    truth_hash = _sha256(args.ground_truth)
    validate_registry_bindings(interps, replay_runs, truth_hash)
    result = score(
        interps, truth,
        load_technique_sequence_records(str(args.attackseq_records)),
        replay_runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
