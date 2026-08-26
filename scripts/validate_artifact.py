"""Validate ATHENA resources and run-derived evidence without placeholders."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from uuid import UUID

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATASETS = ("darpa_e3", "darpa_e5", "optc", "atlas")
TECHNIQUE_RE = re.compile(r"T\d{4}(?:\.\d{3})?")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _label_files(dataset: str) -> list[Path]:
    root = REPO_ROOT / "data" / "annotated_labels" / dataset / "malicious_entities"
    return [path for path in sorted(root.glob("*")) if path.is_file() and any(
        line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
    )]


def _source_linked_technique_schema(root: Path) -> bool:
    from scripts.import_optc_attack_annotations import (
        REVIEWED,
        REVIEWED_EVIDENCE_PINS,
        REVIEWED_SOURCE_PINS,
    )
    annotations_path = root / "source_linked_annotations.jsonl"
    evidence_path = root / "source_records.jsonl"
    mapping_path = root / "mapping_records.jsonl"
    manifest_path = root / "content_manifest.json"
    if not all(path.is_file() for path in (annotations_path, evidence_path, mapping_path, manifest_path)):
        return False
    try:
        annotations = [
            json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evidence_rows = [
            json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mapping_rows = [
            json.loads(line) for line in mapping_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if not annotations or not evidence_rows or len(mapping_rows) != len(annotations) or not isinstance(manifest, dict):
        return False

    if set(REVIEWED) != set(REVIEWED_SOURCE_PINS) or set(REVIEWED) != set(REVIEWED_EVIDENCE_PINS):
        return False

    evidence_by_hash, evidence_tasks = {}, set()
    for row in evidence_rows:
        claimed = str(row.get("record_sha256") or "")
        payload = dict(row)
        payload.pop("record_sha256", None)
        task_id = str(row.get("task_id") or "")
        raw = row.get("raw_event") if isinstance(row.get("raw_event"), dict) else {}
        anchor = row.get("anchor") if isinstance(row.get("anchor"), dict) else {}
        observed_pin = (
            str(row.get("task_record_sha256") or ""),
            str(raw.get("source_record_sha256") or ""),
            str(anchor.get("source_record_sha256") or ""),
        )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", claimed)
            or _canonical_hash(payload) != claimed
            or task_id in evidence_tasks
            or REVIEWED_SOURCE_PINS.get(task_id) != observed_pin
            or REVIEWED_EVIDENCE_PINS.get(task_id) != claimed
            or not REVIEWED.get(task_id)
            or str(raw.get("event_id") or "") != REVIEWED[task_id][0]
        ):
            return False
        if claimed in evidence_by_hash:
            return False
        evidence_by_hash[claimed] = row
        evidence_tasks.add(task_id)
    if evidence_tasks != set(REVIEWED):
        return False

    tactic_path = REPO_ROOT / "data/attack_knowledge/mitre_attack/technique_to_tactic.json"
    tactic_map = _read_json(tactic_path)
    if (
        not isinstance(tactic_map, dict)
        or _sha256(tactic_path) != "6658c7d3fb66b72aa19c6377c902de4a29dcfd77e3e942d4996a0a21f3e9f482"
    ):
        return False
    evidence_sha = _sha256(evidence_path)
    annotation_ids, event_ids = set(), set()
    for row in annotations:
        annotation_id = str(row.get("annotation_id") or "")
        event_id = str(row.get("event_id") or "")
        source_record_sha = str(row.get("source_record_sha256") or "")
        evidence = evidence_by_hash.get(source_record_sha)
        if evidence is None:
            return False
        raw = evidence.get("raw_event") if isinstance(evidence.get("raw_event"), dict) else {}
        anchor = evidence.get("anchor") if isinstance(evidence.get("anchor"), dict) else {}
        task_source = evidence.get("task_source") if isinstance(evidence.get("task_source"), dict) else {}
        task_id = str(evidence.get("task_id") or "")
        expected_event, expected_technique, expected_tactic = REVIEWED.get(task_id, (None, None, None))
        role = str(row.get("anchor_role") or "")
        expected_anchor = str(raw.get(f"{role}ID") or "") if role in {"actor", "object"} else ""
        technique = str(row.get("reference_technique") or "").upper()
        valid_tactics = tactic_map.get(technique)
        if (
            row.get("record_type") != "source_linked_attack_annotation"
            or row.get("annotation_status") != "final_high_confidence"
            or not annotation_id or annotation_id in annotation_ids
            or not event_id or event_id in event_ids
            or row.get("source_id") != annotation_id
            or annotation_id != f"optc-{task_id}"
            or row.get("source_record") != "source_records.jsonl"
            or str(row.get("source_hash") or "") != evidence_sha
            or str((row.get("source_locator") or {}).get("raw_event_id") or "") != event_id
            or str(raw.get("event_id") or "") != event_id
            or event_id != expected_event
            or str(row.get("event_actor") or "") != str(raw.get("actorID") or "")
            or str(row.get("event_object") or "") != str(raw.get("objectID") or "")
            or str(row.get("event_action") or "") != str(raw.get("action") or "")
            or str(row.get("event_timestamp") or "") != str(raw.get("timestamp") or "")
            or str(anchor.get("role") or "") != role
            or str(anchor.get("node_uuid") or "") != str(row.get("anchor_uuid") or "")
            or str(row.get("anchor_uuid") or "") != expected_anchor
            or row.get("scene") != {
                "H201": "day1", "H501": "day2", "H051": "day3",
            }.get(str(row.get("host_id") or ""))
            or str(row.get("source_partition") or "") != {
                "H051": "optc_h051", "H201": "optc_h201", "H501": "optc_h501",
            }.get(str(row.get("host_id") or ""))
            or not str(raw.get("hostname") or "").lower().startswith(
                {"H051": "sysclient0051", "H201": "sysclient0201", "H501": "sysclient0501"}.get(
                    str(row.get("host_id") or ""), "missing-host"
                )
            )
            or str((row.get("source_locator") or {}).get("task_id") or "") != str(evidence.get("task_id") or "")
            or task_source.get("commit") != "64c9f9b2e1a15bf3c2789d89d93dc0724cb0d4fa"
            or task_source.get("archive_sha256") != "b8ddd6d2d82ebbdc637f158cb7e7537c808ddd6968079653405844e1efc83c77"
            or raw.get("archive_sha256") != "9feb73e29c07fd41bbe5670c111bd396bc101991beb39ca3721ddf6565650c2a"
            or not TECHNIQUE_RE.fullmatch(technique)
            or row.get("reference_technique") != expected_technique
            or row.get("reference_tactic") != expected_tactic
            or not isinstance(valid_tactics, list)
            or sorted(row.get("valid_tactics") or []) != sorted(valid_tactics)
            or str(row.get("reference_tactic") or "") not in valid_tactics
        ):
            return False
        annotation_ids.add(annotation_id)
        event_ids.add(event_id)

    annotations_by_id = {str(row["annotation_id"]): row for row in annotations}
    for row in mapping_rows:
        source_id = str(row.get("source_id") or "")
        annotation = annotations_by_id.get(source_id)
        boundary = row.get("boundary") if isinstance(row.get("boundary"), dict) else {}
        mapping_evidence = (
            evidence_by_hash.get(str(annotation.get("source_record_sha256") or ""))
            if annotation is not None else None
        )
        mapping_raw = (
            mapping_evidence.get("raw_event")
            if isinstance(mapping_evidence, dict) and isinstance(mapping_evidence.get("raw_event"), dict)
            else {}
        )
        if annotation is None or (
            row.get("record_type") != "mapping"
            or row.get("dataset") != annotation.get("dataset")
            or row.get("scene") != annotation.get("scene")
            or row.get("host_id") != annotation.get("host_id")
            or row.get("anchor_uuid") != annotation.get("anchor_uuid")
            or row.get("reference_technique") != annotation.get("reference_technique")
            or row.get("reference_tactic") != annotation.get("reference_tactic")
            or row.get("source_record_sha256") != annotation.get("source_record_sha256")
            or boundary.get("event_id") != annotation.get("event_id")
            or boundary.get("anchor") != annotation.get("anchor_uuid")
            or boundary.get("actor") != annotation.get("event_actor")
            or boundary.get("object") != annotation.get("event_object")
            or boundary.get("action") != annotation.get("event_action")
            or boundary.get("timestamp") != annotation.get("event_timestamp")
            or boundary.get("source_event_sha256") != mapping_raw.get("source_record_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(boundary.get("source_event_sha256") or ""))
        ):
            return False

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    converter = manifest.get("converter") if isinstance(manifest.get("converter"), dict) else {}
    aggregate = dict(manifest)
    claimed_aggregate = aggregate.pop("aggregate_sha256", None)
    return bool(
        int(manifest.get("annotation_count", -1)) == len(annotations)
        and outputs.get("source_records.jsonl") == evidence_sha
        and outputs.get("source_linked_annotations.jsonl") == _sha256(annotations_path)
        and outputs.get("mapping_records.jsonl") == _sha256(mapping_path)
        and (manifest.get("attack_taxonomy") or {}).get("sha256") == _sha256(tactic_path)
        and manifest.get("source_commit") == "64c9f9b2e1a15bf3c2789d89d93dc0724cb0d4fa"
        and (manifest.get("source_archives") or {}).get("tasks/tasks.zip")
        == "b8ddd6d2d82ebbdc637f158cb7e7537c808ddd6968079653405844e1efc83c77"
        and (manifest.get("source_archives") or {}).get("labels/malicious.zip")
        == "9feb73e29c07fd41bbe5670c111bd396bc101991beb39ca3721ddf6565650c2a"
        and converter.get("path") == "scripts/import_optc_attack_annotations.py"
        and converter.get("sha256") == _sha256(REPO_ROOT / converter["path"])
        and claimed_aggregate == _canonical_hash(aggregate)
    )


def _darpa_e5_portable_input_from_evidence(row: dict) -> dict:
    """Reconstruct the exact path-sanitized record pinned by the E5 review."""
    raw = row.get("raw_event") if isinstance(row.get("raw_event"), dict) else {}
    anchor = row.get("anchor") if isinstance(row.get("anchor"), dict) else {}
    avro = row.get("official_avro") if isinstance(row.get("official_avro"), dict) else {}
    report = row.get("official_report") if isinstance(row.get("official_report"), dict) else {}
    attack = row.get("attack_taxonomy") if isinstance(row.get("attack_taxonomy"), dict) else {}
    review = row.get("review") if isinstance(row.get("review"), dict) else {}
    occurrence = row.get("occurrence_audit") if isinstance(row.get("occurrence_audit"), dict) else {}
    return {
        "annotation_id": row.get("annotation_id"),
        "status": "exact",
        "evidence_scope": "event_and_report",
        "technique_assignment": "final_event_gt",
        "dataset": "DARPA_TC_E5",
        "platform": row.get("dataset"),
        "attack_id": row.get("source_attack_id"),
        "host_id": raw.get("host_id"),
        "event_uuid": raw.get("raw_event_uuid"),
        "timestamp_nanos": raw.get("timestamp_nanos"),
        "timestamp_utc": raw.get("timestamp"),
        "event_type": raw.get("action"),
        "actor_uuid": raw.get("actorID"),
        "object_uuid": raw.get("primaryObjectID"),
        "object2_uuid": raw.get("object2ID"),
        "anchor_uuid": anchor.get("node_uuid"),
        "anchor_role": anchor.get("role"),
        "anchor_node_type": anchor.get("node_type"),
        "anchor_attributes_raw": anchor.get("attributes"),
        "report": {
            "filename": Path(str(report.get("path") or "")).name,
            "sha256": report.get("sha256"),
            "section": report.get("section"),
            "pdf_pages": report.get("pdf_pages"),
            "pdftotext_line_start": report.get("pdftotext_line_start"),
            "pdftotext_line_end": report.get("pdftotext_line_end"),
            "evidence": report.get("evidence"),
        },
        "attack_technique": {
            "confidence": "high",
            "mapping_basis": review.get("mapping_basis"),
            "tactic": attack.get("tactic"),
            "technique_id": attack.get("technique"),
            "technique_name": attack.get("technique_name"),
        },
        "raw_anchor_occurrence_audit": occurrence,
        "source": {
            "producer": avro.get("producer"),
            "official_avro_filename": avro.get("filename"),
            "official_drive_file_id": avro.get("drive_file_id"),
            "official_avro_sha256": avro.get("sha256"),
            "source_record_ordinal": avro.get("record_ordinal"),
            "raw_event_json_sha256": raw.get("source_record_sha256"),
            "anchor_csv_sha256": anchor.get("source_file_sha256"),
            "anchor_csv": anchor.get("source_path"),
        },
    }


def _darpa_e5_attack_technique_schema(root: Path | None = None) -> tuple[bool, dict]:
    """Validate six reviewed E5 entity labels anchored to exact CDM events.

    The public handler creates one graph edge per CDM predicate endpoint, so
    the graph event ID is the raw CDM UUID plus ``:predicateObject`` (or
    ``:predicateObject2`` for the reviewed CADETS exploit edge).  Local scene
    directory names are deliberately unresolved here; the registry requires
    a unique runtime join by dataset, host, raw event UUID, endpoint and anchor.
    """
    from scripts.import_e5_attack_annotations import (
        EXPECTED_OCCURRENCES,
        EXPECTED_REGISTRY_SHA256,
        EXPECTED_SHARDS,
        EXPECTED_TECHNIQUES,
        PIDSMaker_COMMIT,
        PIDSMaker_REPOSITORY,
        REVIEWED_INPUT_SHA256,
        SCENE_AUDIT,
        TA51_REPORT_PATH,
        TA51_REPORT_SHA256,
        TAXONOMY_SHA256,
    )

    root = root or REPO_ROOT / "data/annotated_labels/darpa_e5/attack_techniques"
    evidence_path = root / "source_records.jsonl"
    annotations_path = root / "source_linked_annotations.jsonl"
    mappings_path = root / "mapping_records.jsonl"
    audit_path = root / "audit.md"
    manifest_path = root / "content_manifest.json"
    paths = (evidence_path, annotations_path, mappings_path, audit_path, manifest_path)
    if not all(path.is_file() for path in paths):
        return False, {"path": str(root), "reason": "missing_e5_annotation_file"}
    try:
        evidence_rows = [
            json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        annotations = [
            json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        mappings = [
            json.loads(line) for line in mappings_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False, {"path": str(root), "reason": "invalid_e5_annotation_json"}
    expected_ids = set(REVIEWED_INPUT_SHA256)
    if not (
        len(evidence_rows) == len(annotations) == len(mappings) == 6
        and isinstance(manifest, dict)
    ):
        return False, {"path": str(root), "reason": "wrong_e5_annotation_count"}

    taxonomy_path = REPO_ROOT / "data/attack_knowledge/mitre_attack/technique_to_tactic.json"
    taxonomy = _read_json(taxonomy_path)
    if not isinstance(taxonomy, dict) or _sha256(taxonomy_path) != TAXONOMY_SHA256:
        return False, {"path": str(taxonomy_path), "reason": "changed_attack_taxonomy"}

    evidence_by_id, evidence_by_hash = {}, {}
    valid = True
    for row in evidence_rows:
        payload = dict(row)
        claimed = str(payload.pop("record_sha256", "")).lower()
        annotation_id = str(row.get("annotation_id") or "")
        raw = row.get("raw_event") if isinstance(row.get("raw_event"), dict) else {}
        anchor = row.get("anchor") if isinstance(row.get("anchor"), dict) else {}
        avro = row.get("official_avro") if isinstance(row.get("official_avro"), dict) else {}
        report = row.get("official_report") if isinstance(row.get("official_report"), dict) else {}
        attack = row.get("attack_taxonomy") if isinstance(row.get("attack_taxonomy"), dict) else {}
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        occurrence = row.get("occurrence_audit") if isinstance(row.get("occurrence_audit"), dict) else {}
        role = str(anchor.get("role") or "")
        endpoint = "predicateObject2" if role == "object2" else "predicateObject"
        raw_uuid = str(raw.get("raw_event_uuid") or "")
        graph_event_id = f"{raw_uuid}:{endpoint}"
        selected_anchor = {
            "actor": raw.get("actorID"),
            "object": raw.get("primaryObjectID"),
            "object2": raw.get("object2ID"),
        }.get(role)
        expected_technique = EXPECTED_TECHNIQUES.get(annotation_id)
        source_path = str(anchor.get("source_path") or "")
        report_path = str(report.get("path") or "")
        valid = valid and bool(
            row.get("record_type") == "darpa_e5_manual_annotation_evidence"
            and row.get("schema") == "athena.darpa_e5.attack_annotation_evidence.v1"
            and annotation_id in expected_ids and annotation_id not in evidence_by_id
            and re.fullmatch(r"[0-9a-f]{64}", claimed)
            and _canonical_hash(payload) == claimed
            and row.get("input_record_sha256") == REVIEWED_INPUT_SHA256.get(annotation_id)
            and _canonical_hash(_darpa_e5_portable_input_from_evidence(row))
            == REVIEWED_INPUT_SHA256.get(annotation_id)
            and row.get("dataset") in {"cadets5", "clearscope5", "theia5", "trace5"}
            and re.fullmatch(r"[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}", raw_uuid)
            and raw.get("event_id") == graph_event_id
            and raw.get("graph_event_id") == graph_event_id
            and anchor.get("node_uuid") == selected_anchor
            and (role != "object2" or attack.get("technique") == "T1190")
            and (role == "object2" or endpoint == "predicateObject")
            and avro.get("sha256") == EXPECTED_SHARDS.get(avro.get("filename"))
            and re.fullmatch(r"[0-9a-f]{64}", str(raw.get("source_record_sha256") or ""))
            and report.get("source_repository") == PIDSMaker_REPOSITORY
            and report.get("source_commit") == PIDSMaker_COMMIT
            and report_path == TA51_REPORT_PATH and not Path(report_path).is_absolute()
            and report.get("sha256") == TA51_REPORT_SHA256
            and source_path.startswith("Ground_Truth/") and not Path(source_path).is_absolute()
            and attack.get("taxonomy_sha256") == TAXONOMY_SHA256
            and expected_technique == (attack.get("technique"), attack.get("tactic"))
            and attack.get("tactic") in taxonomy.get(attack.get("technique"), [])
            and review.get("status") == "final_high_confidence"
            and review.get("evidence_scope") == "event_and_report"
            and occurrence.get("selected_event_matches") == 1
            and occurrence.get("events_containing_anchor") == EXPECTED_OCCURRENCES.get(annotation_id)
        )
        evidence_by_id[annotation_id] = row
        evidence_by_hash[claimed] = row
    valid = valid and set(evidence_by_id) == expected_ids and len(evidence_by_hash) == 6

    evidence_file_sha = _sha256(evidence_path)
    annotations_by_id = {}
    for row in annotations:
        annotation_id = str(row.get("annotation_id") or "")
        evidence = evidence_by_hash.get(str(row.get("source_record_sha256") or ""))
        raw = evidence.get("raw_event", {}) if isinstance(evidence, dict) else {}
        anchor = evidence.get("anchor", {}) if isinstance(evidence, dict) else {}
        attack = evidence.get("attack_taxonomy", {}) if isinstance(evidence, dict) else {}
        locator = row.get("source_locator") if isinstance(row.get("source_locator"), dict) else {}
        valid = valid and bool(
            evidence is not None and annotation_id == evidence.get("annotation_id")
            and annotation_id not in annotations_by_id
            and row.get("record_type") == "source_linked_attack_annotation"
            and row.get("annotation_status") == "final_high_confidence"
            and row.get("source_id") == annotation_id
            and row.get("dataset") == evidence.get("dataset")
            and row.get("scene") == "*"
            and row.get("scene_resolution") == "runtime_unique_raw_event"
            and row.get("source_attack_id") == evidence.get("source_attack_id")
            and row.get("host_id") == raw.get("host_id")
            and row.get("anchor_uuid") == anchor.get("node_uuid")
            and row.get("anchor_role") == anchor.get("role")
            and row.get("event_id") == raw.get("graph_event_id")
            and row.get("raw_event_uuid") == raw.get("raw_event_uuid")
            and row.get("event_actor") == raw.get("actorID")
            and row.get("event_object") == raw.get("objectID")
            and row.get("event_action") == raw.get("action")
            and row.get("event_timestamp") == raw.get("timestamp")
            and row.get("reference_technique") == attack.get("technique")
            and row.get("reference_tactic") == attack.get("tactic")
            and sorted(row.get("valid_tactics") or []) == sorted(taxonomy.get(attack.get("technique"), []))
            and row.get("source_record") == "source_records.jsonl"
            and row.get("source_hash") == evidence_file_sha
            and locator.get("raw_event_id") == raw.get("raw_event_uuid")
            and locator.get("graph_event_id") == raw.get("graph_event_id")
            and locator.get("official_avro_filename") == evidence.get("official_avro", {}).get("filename")
            and locator.get("official_avro_sha256") == evidence.get("official_avro", {}).get("sha256")
            and locator.get("raw_event_json_sha256") == raw.get("source_record_sha256")
            and locator.get("report_sha256") == TA51_REPORT_SHA256
        )
        annotations_by_id[annotation_id] = row
    valid = valid and set(annotations_by_id) == expected_ids

    mapping_ids = set()
    for row in mappings:
        annotation_id = str(row.get("source_id") or "")
        annotation = annotations_by_id.get(annotation_id)
        evidence = evidence_by_id.get(annotation_id)
        raw = evidence.get("raw_event", {}) if isinstance(evidence, dict) else {}
        boundary = row.get("boundary") if isinstance(row.get("boundary"), dict) else {}
        valid = valid and bool(
            annotation is not None and annotation_id not in mapping_ids
            and row.get("record_type") == "mapping"
            and row.get("dataset") == annotation.get("dataset")
            and row.get("scene") == "*"
            and row.get("scene_resolution") == "runtime_unique_raw_event"
            and row.get("source_attack_id") == annotation.get("source_attack_id")
            and row.get("host_id") == annotation.get("host_id")
            and row.get("anchor_uuid") == annotation.get("anchor_uuid")
            and row.get("reference_technique") == annotation.get("reference_technique")
            and row.get("reference_tactic") == annotation.get("reference_tactic")
            and row.get("source_record") == "source_records.jsonl"
            and row.get("source_hash") == evidence_file_sha
            and row.get("source_record_sha256") == annotation.get("source_record_sha256")
            and boundary.get("snapshot") is None
            and boundary.get("event_id") == raw.get("graph_event_id")
            and boundary.get("raw_event_uuid") == raw.get("raw_event_uuid")
            and boundary.get("anchor") == annotation.get("anchor_uuid")
            and boundary.get("anchor_role") == annotation.get("anchor_role")
            and boundary.get("actor") == raw.get("actorID")
            and boundary.get("object") == raw.get("objectID")
            and boundary.get("action") == raw.get("action")
            and boundary.get("timestamp") == raw.get("timestamp")
            and boundary.get("timestamp_nanos") == raw.get("timestamp_nanos")
            and boundary.get("source_event_sha256") == raw.get("source_record_sha256")
        )
        mapping_ids.add(annotation_id)
    valid = valid and mapping_ids == expected_ids

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    converter = manifest.get("converter") if isinstance(manifest.get("converter"), dict) else {}
    converter_path = REPO_ROOT / str(converter.get("path") or "")
    generator_sha = str((manifest.get("source_registry_generator") or {}).get("sha256") or "")
    occurrence_sha = str((manifest.get("raw_occurrence_counter") or {}).get("sha256") or "")
    aggregate = dict(manifest)
    claimed_aggregate = aggregate.pop("aggregate_sha256", None)
    manifest_ok = bool(
        manifest.get("schema") == "athena.darpa_e5.final_attack_annotations.v1"
        and manifest.get("dataset") == "darpa_e5"
        and manifest.get("annotation_status") == "final_high_confidence"
        and manifest.get("annotation_count") == 6
        and manifest.get("exact_attack_scene_count") == 4
        and manifest.get("incomplete_attack_scene_count") == 4
        and manifest.get("paper_platforms") == ["cadets5", "clearscope5", "theia5", "trace5"]
        and manifest.get("paper_profile_pidsmaker_attack_sources") == 8
        and manifest.get("five_directions_in_paper_profile") is False
        and manifest.get("scene_audit") == SCENE_AUDIT
        and manifest.get("technique_counts") == dict(sorted(Counter(
            row["reference_technique"] for row in annotations
        ).items()))
        and manifest.get("official_shards") == EXPECTED_SHARDS
        and (manifest.get("input_registry") or {}) == {
            "sha256": EXPECTED_REGISTRY_SHA256, "rows": 6,
        }
        and (manifest.get("official_report") or {}).get("repository") == PIDSMaker_REPOSITORY
        and (manifest.get("official_report") or {}).get("commit") == PIDSMaker_COMMIT
        and (manifest.get("official_report") or {}).get("path") == TA51_REPORT_PATH
        and (manifest.get("official_report") or {}).get("sha256") == TA51_REPORT_SHA256
        and (manifest.get("attack_taxonomy") or {}).get("sha256") == TAXONOMY_SHA256
        and re.fullmatch(r"[0-9a-f]{64}", generator_sha)
        and re.fullmatch(r"[0-9a-f]{64}", occurrence_sha)
        and outputs.get("source_records.jsonl") == evidence_file_sha
        and outputs.get("source_linked_annotations.jsonl") == _sha256(annotations_path)
        and outputs.get("mapping_records.jsonl") == _sha256(mappings_path)
        and outputs.get("audit.md") == _sha256(audit_path)
        and converter.get("path") == "scripts/import_e5_attack_annotations.py"
        and converter_path.is_file() and converter.get("sha256") == _sha256(converter_path)
        and claimed_aggregate == _canonical_hash(aggregate)
    )
    valid = bool(valid and manifest_ok)
    return valid, {
        "path": str(root),
        "annotations": len(annotations),
        "exact_scenes": sum(row.get("status") == "exact" for row in SCENE_AUDIT),
        "incomplete_scenes": sum(row.get("status") == "incomplete" for row in SCENE_AUDIT),
        "technique_counts": dict(Counter(row.get("reference_technique") for row in annotations)),
        "runtime_scene_resolution": "unique dataset+host+graph_event_id+anchor join",
        "manifest_ok": manifest_ok,
    }


def _atlas_source_linked_registry(root: Path | None = None) -> tuple[bool, dict]:
    """Validate the official-v1 endpoint registry and keep mappings non-final."""
    root = root or REPO_ROOT / "data/annotated_labels/atlas/source_linked"
    records_path = root / "atlas_v1_source_link_endpoints.jsonl"
    audit_path = root / "atlas_v1_source_link_audit.json"
    manifest_path = root / "atlas_v1_source_link_manifest.json"
    paths = (records_path, audit_path, manifest_path)
    if not all(path.is_file() for path in paths):
        return False, {"path": str(root), "reason": "missing_source_linked_files"}
    raw_records = records_path.read_text(encoding="utf-8")
    raw_audit = audit_path.read_text(encoding="utf-8")
    if any(token in raw_records + raw_audit for token in ("/Users/", "/tmp/")):
        return False, {"path": str(root), "reason": "nonportable_absolute_path"}
    try:
        records = [json.loads(line) for line in raw_records.splitlines() if line.strip()]
        audit = json.loads(raw_audit)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False, {"path": str(root), "reason": "invalid_json"}
    official_scenes = {"S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4", "M5", "M6"}
    record_pairs, source_ids = [], set()
    valid = bool(records)
    for row in records:
        payload = dict(row)
        claimed = str(payload.pop("derived_sha256", ""))
        source_id = str(row.get("source_id") or "")
        boundary = row.get("boundary") if isinstance(row.get("boundary"), dict) else {}
        source = row.get("source") if isinstance(row.get("source"), dict) else {}
        candidates = row.get("attack_mapping_candidates")
        valid = valid and bool(
            row.get("schema") == "athena.atlas_v1.malicious_endpoint_join.v1"
            and row.get("record_type") == "malicious_endpoint_join"
            and row.get("dataset") == "atlas"
            and row.get("scene") in official_scenes
            and row.get("host") in {"H1", "H2"}
            and source_id.startswith("atlas-v1:") and source_id not in source_ids
            and re.fullmatch(r"[0-9a-f]{64}", claimed)
            and _canonical_hash(payload) == claimed
            and row.get("resolved_anchor") == boundary.get("anchor")
            and row.get("matched_labels")
            and row.get("official_labels")
            and str(boundary.get("event_id") or "")
            and re.fullmatch(r"[0-9a-f]{64}", str(boundary.get("event_sha256") or ""))
            and str(source.get("zip") or "").endswith(".zip")
            and not Path(str(source.get("zip") or "")).is_absolute()
            and not Path(str(source.get("preprocessed_path") or "")).is_absolute()
            and not Path(str(source.get("malicious_labels_path") or "")).is_absolute()
            and all(re.fullmatch(r"[0-9a-f]{64}", str(source.get(key) or "")) for key in (
                "zip_sha256", "preprocessed_sha256", "malicious_labels_sha256",
            ))
            and row.get("mapping_status")
            == "human_candidates_separate_from_official_endpoint_ground_truth"
            and isinstance(candidates, list) and candidates
            and all(
                candidate.get("status") == "candidate_not_official_atlas_annotation"
                and candidate.get("confidence") == "high"
                and candidate.get("kind") in {"technique", "tactic"}
                and re.fullmatch(r"T(?:A)?\d{4}(?:\.\d{3})?", str(candidate.get("id") or ""))
                for candidate in candidates if isinstance(candidate, dict)
            )
            and len(candidates) == sum(isinstance(candidate, dict) for candidate in candidates)
        )
        source_ids.add(source_id)
        record_pairs.append({"source_id": source_id, "derived_sha256": claimed})

    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    files_ok = all(
        name in files
        and _sha256(root / name) == str(files[name].get("sha256") or "")
        and (root / name).stat().st_size == int(files[name].get("size_bytes", -1))
        for name in (audit_path.name, records_path.name)
    )
    aggregate = hashlib.sha256(b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        for row in record_pairs
    )).hexdigest()
    parser_sources = manifest.get("parser_sources") if isinstance(manifest.get("parser_sources"), dict) else {}
    parsers_ok = bool(parser_sources) and all(
        (REPO_ROOT / rel).is_file() and _sha256(REPO_ROOT / rel) == digest
        for rel, digest in parser_sources.items()
    )
    generator = manifest.get("generator") if isinstance(manifest.get("generator"), dict) else {}
    generator_path = REPO_ROOT / "scripts" / str(generator.get("filename") or "")
    generator_ok = generator_path.is_file() and _sha256(generator_path) == generator.get("sha256")
    summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
    audit_candidates = (audit.get("attack_evidence") or {}).get("scenarios", [])
    audit_ok = bool(
        audit.get("schema") == "athena.atlas_v1.source_link_audit.v1"
        and int(summary.get("matched_endpoint_join_count", -1)) == len(records)
        and int(summary.get("scenario_count", -1)) == 10
        and int(summary.get("host_stream_count", -1)) == 16
        and len(audit.get("cases") or []) == 16
        and {row.get("scenario") for row in audit_candidates} == official_scenes
        and all(
            candidate.get("status") == "candidate_not_official_atlas_annotation"
            for scenario in audit_candidates
            for candidate in scenario.get("attack_mapping_candidates", [])
        )
    )
    source = manifest.get("source_repository") if isinstance(manifest.get("source_repository"), dict) else {}
    manifest_ok = bool(
        manifest.get("schema") == "athena.atlas_v1.source_link_manifest.v1"
        and int(manifest.get("record_count", -1)) == len(records) == 228
        and set(manifest.get("scenarios") or []) == official_scenes
        and len(manifest.get("hosts") or []) == 16
        and manifest.get("records") == record_pairs
        and manifest.get("aggregate_records_sha256") == aggregate
        and source.get("url") == "https://github.com/purseclab/ATLAS.git"
        and source.get("commit") == "e46096d1947e4f059e73a0ac2b9a9707812fd4bc"
        and len(manifest.get("archives") or {}) == 10
    )
    ok = bool(valid and files_ok and parsers_ok and generator_ok and audit_ok and manifest_ok)
    return ok, {
        "path": str(root), "records": len(records), "scenarios": len(official_scenes),
        "host_streams": len(manifest.get("hosts") or []), "files_ok": files_ok,
        "parsers_ok": parsers_ok, "generator_ok": generator_ok,
        "audit_ok": audit_ok, "manifest_ok": manifest_ok,
    }


def _atlas_attack_technique_schema(root: Path | None = None) -> tuple[bool, dict]:
    """Validate reviewed ATLAS-v1 mappings against the exact endpoint registry."""
    from scripts.import_atlas_v1_attack_annotations import (
        ATLAS_COMMIT, ATLAS_PAPER_SHA256, ATLAS_PAPER_URL,
        AUDIT_GLOBAL_FAMILY_BASE, BOUNDARY_PINS, PA_OCCURRENCE_COUNTS,
        REVIEWED, TAXONOMY_SHA256, _boundary_pin, _family_boundary,
    )

    root = root or REPO_ROOT / "data/annotated_labels/atlas/attack_techniques"
    evidence_path = root / "source_records.jsonl"
    annotations_path = root / "source_linked_annotations.jsonl"
    mappings_path = root / "mapping_records.jsonl"
    manifest_path = root / "content_manifest.json"
    if not all(path.is_file() for path in (
        evidence_path, annotations_path, mappings_path, manifest_path,
    )):
        return False, {"path": str(root), "reason": "missing_final_mapping_files"}
    try:
        evidence_rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        annotations = [json.loads(line) for line in annotations_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mappings = [json.loads(line) for line in mappings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False, {"path": str(root), "reason": "invalid_final_mapping_json"}

    registry_meta = manifest.get("endpoint_registry") if isinstance(manifest.get("endpoint_registry"), dict) else {}
    registry_path = root / str(registry_meta.get("path") or "")
    if not registry_path.is_file():
        return False, {"path": str(root), "reason": "missing_endpoint_registry"}
    try:
        endpoint_rows = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError):
        return False, {"path": str(root), "reason": "invalid_endpoint_registry"}
    endpoints = {str(row.get("source_id") or ""): row for row in endpoint_rows}
    if len(endpoints) != len(endpoint_rows):
        return False, {"path": str(root), "reason": "duplicate_endpoint_source_id"}

    evidence_by_hash, evidence_by_endpoint = {}, {}
    valid = bool(evidence_rows) and len(evidence_rows) == len(REVIEWED)
    for evidence in evidence_rows:
        payload = dict(evidence)
        claimed = str(payload.pop("record_sha256", ""))
        registry = evidence.get("endpoint_registry") if isinstance(evidence.get("endpoint_registry"), dict) else {}
        endpoint = evidence.get("endpoint_record") if isinstance(evidence.get("endpoint_record"), dict) else {}
        paper = evidence.get("paper_source") if isinstance(evidence.get("paper_source"), dict) else {}
        review = evidence.get("review") if isinstance(evidence.get("review"), dict) else {}
        taxonomy = evidence.get("attack_taxonomy") if isinstance(evidence.get("attack_taxonomy"), dict) else {}
        occurrence_audit = evidence.get("occurrence_audit") if isinstance(evidence.get("occurrence_audit"), dict) else {}
        selected = evidence.get("selected_boundary") if isinstance(evidence.get("selected_boundary"), dict) else {}
        source_id = str(registry.get("source_id") or "")
        source_row = endpoints.get(source_id)
        boundary = endpoint.get("boundary") if isinstance(endpoint.get("boundary"), dict) else {}
        reviewed = REVIEWED.get(source_id)
        if reviewed is None:
            valid = False
            continue
        scene, host, anchor, anchor_type, action, feature, technique, tactic = reviewed
        expected_criterion = (
            "table2_pl_plus_initial_host_malicious_web_object"
            if feature == "PL" else
            "table2_pa_plus_first_browser_delivery_write"
        )
        if feature == "PL":
            occurrence_ok = bool(
                occurrence_audit.get("schema") == "athena.atlas_v1.unique_endpoint_occurrence.v1"
                and occurrence_audit.get("occurrence_count") == 1
                and selected == _family_boundary(scene, boundary)
                and int(endpoint.get("matched_event_occurrence_count", -1)) == 1
            )
        else:
            occurrences = occurrence_audit.get("occurrences") if isinstance(occurrence_audit.get("occurrences"), list) else []
            delivery = occurrence_audit.get("delivery_candidates") if isinstance(occurrence_audit.get("delivery_candidates"), list) else []
            expected_total, expected_delivery = PA_OCCURRENCE_COUNTS[scene]
            calculated_delivery = [
                row for row in occurrences
                if row.get("action") == "file_write"
                and row.get("endpoint_role") == "target"
                and row.get("peer_type") == "process"
                and "firefox.exe_" in str(row.get("peer") or "")
            ]
            calculated_delivery.sort(key=lambda row: (
                float(row.get("event_timestamp", 0.0)), int(row.get("event_order", -1)),
                str(row.get("event_id") or ""),
            ))
            occurrence_ok = bool(
                occurrence_audit.get("schema") == "athena.atlas_v1.attachment_occurrence_audit.v1"
                and int(occurrence_audit.get("occurrence_count", -1)) == len(occurrences) == expected_total
                and int(occurrence_audit.get("delivery_candidate_count", -1)) == len(delivery) == expected_delivery
                and delivery == calculated_delivery
                and delivery and selected == delivery[0]
                and all(row.get("anchor") == anchor for row in occurrences)
                and occurrence_audit.get("source_preprocessed_path") == (endpoint.get("source") or {}).get("preprocessed_path")
                and occurrence_audit.get("source_preprocessed_sha256") == (endpoint.get("source") or {}).get("preprocessed_sha256")
                and all(
                    int(row.get("source_global_snapshot", -1)) - int(row.get("snapshot", -2))
                    == AUDIT_GLOBAL_FAMILY_BASE[scene[0]]
                    for row in occurrences
                )
            )
        valid = valid and bool(
            evidence.get("record_type") == "atlas_v1_manual_annotation_evidence"
            and evidence.get("schema") == "athena.atlas_v1.attack_annotation_evidence.v1"
            and re.fullmatch(r"[0-9a-f]{64}", claimed)
            and _canonical_hash(payload) == claimed
            and claimed not in evidence_by_hash
            and source_id not in evidence_by_endpoint
            and source_row == endpoint
            and registry.get("path") == "../source_linked/atlas_v1_source_link_endpoints.jsonl"
            and registry.get("sha256") == _sha256(registry_path)
            and registry.get("derived_sha256") == endpoint.get("derived_sha256")
            and (endpoint.get("scene"), endpoint.get("host"), boundary.get("anchor"),
                 boundary.get("anchor_type"), boundary.get("action"))
                == (scene, host, anchor, anchor_type, action)
            and feature in (endpoint.get("paper_attack_features") or [])
            and paper.get("url") == ATLAS_PAPER_URL
            and paper.get("sha256") == ATLAS_PAPER_SHA256
            and paper.get("feature") == feature
            and paper.get("feature_reference") == "Table 2, PDF page 12, proceedings page 3015"
            and review.get("criterion") == expected_criterion
            and bool(review.get("basis"))
            and occurrence_ok
            and _boundary_pin(selected) == BOUNDARY_PINS[source_id]
            and taxonomy.get("sha256") == TAXONOMY_SHA256
            and taxonomy.get("technique") == technique
            and taxonomy.get("tactic") == tactic
        )
        evidence_by_hash[claimed] = evidence
        evidence_by_endpoint[source_id] = evidence

    tactic_path = REPO_ROOT / "data/attack_knowledge/mitre_attack/technique_to_tactic.json"
    tactic_map = _read_json(tactic_path)
    valid = valid and bool(
        isinstance(tactic_map, dict) and _sha256(tactic_path) == TAXONOMY_SHA256
        and _sha256(registry_path) == registry_meta.get("sha256")
        and int(registry_meta.get("record_count", -1)) == len(endpoint_rows) == 228
    )
    evidence_file_sha = _sha256(evidence_path)
    annotation_by_id, annotation_sources, events = {}, set(), set()
    for annotation in annotations:
        annotation_id = str(annotation.get("annotation_id") or "")
        evidence = evidence_by_hash.get(str(annotation.get("source_record_sha256") or ""))
        endpoint = evidence.get("endpoint_record") if isinstance(evidence, dict) else {}
        boundary = evidence.get("selected_boundary") if isinstance(evidence, dict) and isinstance(evidence.get("selected_boundary"), dict) else {}
        endpoint_source = str((evidence.get("endpoint_registry") or {}).get("source_id") or "") if evidence else ""
        reviewed = REVIEWED.get(endpoint_source)
        if reviewed is None:
            valid = False
            continue
        scene, host, anchor, anchor_type, _endpoint_action, _feature, technique, tactic = reviewed
        valid_tactics = tactic_map.get(technique) if isinstance(tactic_map, dict) else None
        valid = valid and bool(
            annotation.get("record_type") == "source_linked_attack_annotation"
            and annotation.get("annotation_status") == "final_high_confidence"
            and annotation_id and annotation_id not in annotation_by_id
            and annotation.get("source_id") == annotation_id
            and endpoint_source not in annotation_sources
            and annotation.get("dataset") == "atlas"
            and (annotation.get("scene"), annotation.get("host_id"), annotation.get("anchor_uuid"),
                 annotation.get("anchor_type"), annotation.get("event_action"))
                == (scene, host, anchor, anchor_type, boundary.get("action"))
            and int(annotation.get("snapshot", -1)) == int(boundary.get("snapshot", -2))
            and int(annotation.get("source_global_snapshot", -1))
                == int(boundary.get("source_global_snapshot", -2))
            and annotation.get("event_id") == boundary.get("event_id")
            and annotation.get("event_timestamp") == boundary.get("event_timestamp")
            and annotation.get("source_event_sha256") == boundary.get("event_sha256")
            and annotation.get("anchor_role") == boundary.get("endpoint_role")
            and annotation.get("reference_technique") == technique
            and annotation.get("reference_tactic") == tactic
            and annotation.get("valid_tactics") == valid_tactics
            and annotation.get("source_record") == "source_records.jsonl"
            and annotation.get("source_hash") == evidence_file_sha
            and (annotation.get("source_locator") or {}).get("endpoint_source_id") == endpoint_source
            and (annotation.get("source_locator") or {}).get("raw_event_id") == boundary.get("event_id")
            and (annotation.get("source_locator") or {}).get("paper_sha256") == ATLAS_PAPER_SHA256
            and annotation.get("judgment_basis") == evidence.get("review")
            and annotation.get("event_id") not in events
        )
        annotation_by_id[annotation_id] = annotation
        annotation_sources.add(endpoint_source)
        events.add(annotation.get("event_id"))
    valid = valid and annotation_sources == set(REVIEWED)

    mapping_ids = set()
    for mapping in mappings:
        source_id = str(mapping.get("source_id") or "")
        annotation = annotation_by_id.get(source_id)
        evidence = evidence_by_hash.get(str(mapping.get("source_record_sha256") or ""))
        source_boundary = evidence.get("selected_boundary") if isinstance(evidence, dict) and isinstance(evidence.get("selected_boundary"), dict) else {}
        boundary = mapping.get("boundary") if isinstance(mapping.get("boundary"), dict) else {}
        valid = valid and bool(
            annotation is not None and source_id not in mapping_ids
            and mapping.get("record_type") == "mapping"
            and mapping.get("dataset") == "atlas"
            and mapping.get("scene") == annotation.get("scene")
            and mapping.get("host_id") == annotation.get("host_id")
            and mapping.get("anchor_uuid") == annotation.get("anchor_uuid")
            and mapping.get("reference_technique") == annotation.get("reference_technique")
            and mapping.get("reference_tactic") == annotation.get("reference_tactic")
            and mapping.get("source_record") == "source_records.jsonl"
            and mapping.get("source_hash") == evidence_file_sha
            and boundary.get("snapshot") == source_boundary.get("snapshot")
            and boundary.get("source_global_snapshot") == source_boundary.get("source_global_snapshot")
            and boundary.get("event_id") == source_boundary.get("event_id")
            and boundary.get("anchor") == source_boundary.get("anchor")
            and boundary.get("action") == source_boundary.get("action")
            and boundary.get("timestamp") == source_boundary.get("event_timestamp")
            and boundary.get("event_sha256") == source_boundary.get("event_sha256")
            and boundary.get("source_event_sha256") == source_boundary.get("event_sha256")
        )
        mapping_ids.add(source_id)
    valid = valid and mapping_ids == set(annotation_by_id) and len(mappings) == len(annotations)

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    converter = manifest.get("converter") if isinstance(manifest.get("converter"), dict) else {}
    converter_path = REPO_ROOT / str(converter.get("path") or "")
    aggregate = dict(manifest)
    claimed_aggregate = aggregate.pop("aggregate_sha256", None)
    manifest_ok = bool(
        manifest.get("schema") == "athena.atlas_v1.final_attack_annotations.v1"
        and manifest.get("dataset") == "atlas"
        and manifest.get("annotation_status") == "final_high_confidence"
        and int(manifest.get("annotation_count", -1)) == len(annotations) == 10
        and manifest.get("source_commit") == ATLAS_COMMIT
        and (manifest.get("paper_source") or {}).get("url") == ATLAS_PAPER_URL
        and (manifest.get("paper_source") or {}).get("sha256") == ATLAS_PAPER_SHA256
        and (manifest.get("attack_taxonomy") or {}).get("sha256") == TAXONOMY_SHA256
        and manifest.get("technique_counts") == {"T1566.001": 4, "T1566.002": 6}
        and (manifest.get("snapshot_coordinates") or {}).get("audit_global_family_base")
        == AUDIT_GLOBAL_FAMILY_BASE
        and (manifest.get("review_scope") or {}).get("pa_occurrence_counts") == {
            scene: {"all": counts[0], "browser_delivery_candidates": counts[1]}
            for scene, counts in sorted(PA_OCCURRENCE_COUNTS.items())
        }
        and outputs.get(evidence_path.name) == evidence_file_sha
        and outputs.get(annotations_path.name) == _sha256(annotations_path)
        and outputs.get(mappings_path.name) == _sha256(mappings_path)
        and converter_path.is_file() and _sha256(converter_path) == converter.get("sha256")
        and claimed_aggregate == _canonical_hash(aggregate)
    )
    ok = bool(valid and manifest_ok)
    return ok, {
        "path": str(root), "annotations": len(annotations),
        "technique_counts": dict(Counter(row.get("reference_technique") for row in annotations)),
        "endpoint_registry_bound": annotation_sources == set(REVIEWED),
        "manifest_ok": manifest_ok,
    }


def _technique_schema(dataset: str) -> tuple[bool, int]:
    root = REPO_ROOT / "data" / "annotated_labels" / dataset / "attack_techniques"
    files = sorted(root.glob("*.json"))
    valid = 0
    for path in files:
        payload = _read_json(path)
        entities = payload.get("entities") if isinstance(payload, dict) else None
        if isinstance(entities, dict) and entities and all(
            isinstance(entity, dict)
            and TECHNIQUE_RE.fullmatch(str(entity.get("technique", "")).strip().upper())
            and bool(str(entity.get("tactic", "")).strip())
            for uuid, entity in entities.items() if str(uuid).strip()
        ) and len(entities) == sum(bool(str(uuid).strip()) for uuid in entities):
            valid += 1
    json_ok = bool(files) and valid == len(files)
    if dataset == "atlas":
        source_ok, detail = _atlas_attack_technique_schema(root)
        return json_ok or source_ok, valid + int(source_ok)
    if dataset == "darpa_e5":
        source_ok, detail = _darpa_e5_attack_technique_schema(root)
        return json_ok or source_ok, valid + int(source_ok)
    source_ok = _source_linked_technique_schema(root)
    return json_ok or source_ok, valid + int(source_ok)


def _pidsmaker_registry(dataset: str) -> tuple[bool, dict]:
    root = REPO_ROOT / "data" / "annotated_labels" / dataset / "malicious_entities"
    manifest_path = root / "content_manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        return False, {"path": str(manifest_path), "reason": "missing_or_invalid"}
    expected_counts = (
        {"host_0051.txt": 114, "host_0201.txt": 2905, "host_0501.txt": 749}
        if dataset == "optc" else None
    )
    outputs = manifest.get("output_files") if isinstance(manifest.get("output_files"), list) else []
    output_index = {str(row.get("path")): row for row in outputs if isinstance(row, dict)}
    output_ok = bool(outputs)
    output_sets = {}
    for filename, row in output_index.items():
        path = root / filename
        output_ok = output_ok and path.is_file() and _sha256(path) == str(row.get("sha256"))
        output_ok = output_ok and len(path.read_text(encoding="utf-8").splitlines()) == int(row.get("count", -1))
        values = {
            value.strip().lower() for value in path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        } if path.is_file() else set()
        try:
            output_ok = output_ok and all(str(UUID(value)) == value for value in values)
        except ValueError:
            output_ok = False
        output_sets[filename] = values
    if expected_counts is not None:
        output_ok = output_ok and set(output_index) == set(expected_counts)
        output_ok = output_ok and all(
            int(output_index[name].get("count", -1)) == count for name, count in expected_counts.items()
        )
        output_ok = output_ok and {path.name for path in root.glob("host_*.txt")} == set(expected_counts)
    entities = manifest.get("entity_records") if isinstance(manifest.get("entity_records"), dict) else {}
    entities_path = root / str(entities.get("path") or "")
    entity_ok = entities_path.is_file() and _sha256(entities_path) == str(entities.get("sha256"))
    entity_rows = []
    if entity_ok:
        try:
            entity_rows = [json.loads(line) for line in entities_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except json.JSONDecodeError:
            entity_ok = False
    entity_ok = entity_ok and len(entity_rows) == int(entities.get("count", -1))
    entity_ok = entity_ok and all(
        row.get("record_type") == "malicious_entity"
        and row.get("pidsmaker_export_index") is not None
        and not row.get("event_id")
        and row.get("source_commit") == manifest.get("source_commit")
        for row in entity_rows
    )
    joinable_by_attack = {}
    joinable_by_host = {}
    for row in entity_rows:
        node_id = str(row.get("node_id_canonical") or "").lower()
        try:
            entity_ok = entity_ok and str(UUID(node_id)) == node_id
        except ValueError:
            entity_ok = False
        if not row.get("joinable"):
            continue
        joinable_by_attack.setdefault(str(row.get("attack_id") or ""), set()).add(node_id)
        if row.get("host_id"):
            joinable_by_host.setdefault(str(row["host_id"]), set()).add(node_id)
    sources = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    for source in sources:
        filename = str(source.get("output") or "")
        expected_set = (
            joinable_by_host.get(str(source.get("host_id") or ""), set())
            if dataset == "optc" else
            joinable_by_attack.get(str(source.get("attack_id") or ""), set())
        )
        output_ok = output_ok and output_sets.get(filename) == expected_set
    aggregate = dict(manifest)
    claimed_aggregate = aggregate.pop("aggregate_sha256", None)
    aggregate_ok = claimed_aggregate == _canonical_hash(aggregate)
    source_ok = (
        manifest.get("source_repository") == "https://github.com/ubc-provenance/PIDSMaker.git"
        and manifest.get("source_commit") == "32602734bc9f896be5fc0f03f0a185c967cd6624"
        and (manifest.get("source_license") or {}).get("sha256")
        == "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    )
    return bool(source_ok and output_ok and entity_ok and aggregate_ok), {
        "path": str(manifest_path), "sources": len(manifest.get("sources") or []),
        "entities": len(entity_rows), "outputs": {key: row.get("count") for key, row in output_index.items()},
    }


def _runtime_checks(path: Path, dataset: str):
    payload = _read_json(path)
    base = f"{dataset} runtime report"
    if not isinstance(payload, dict):
        return [(base, False, {"path": str(path), "reason": "missing_or_invalid"})]
    hosts = [str(value) for value in payload.get("hosts", []) if str(value)]
    split = payload.get("split") if isinstance(payload.get("split"), dict) else {}
    train = {str(value) for value in split.get("train_snapshots", [])}
    test = {str(value) for value in split.get("test_snapshots", [])}
    join = payload.get("label_join") if isinstance(payload.get("label_join"), dict) else {}
    rows = [
        (f"{base}: stable hosts", bool(hosts) and len(hosts) == len(set(hosts)), {"hosts": hosts}),
        (f"{base}: disjoint nonempty split", bool(train) and bool(test) and train.isdisjoint(test), {
            "train": len(train), "test": len(test), "overlap": sorted(train & test),
        }),
        (f"{base}: label join", int(join.get("matched", 0) or 0) > 0 and int(join.get("unmatched", 0) or 0) == 0, join),
    ]
    if dataset == "atlas":
        rows.append((
            f"{base}: official ATLAS v1 same-case coverage",
            payload.get("label_source") == "official-atlas-v1"
            and payload.get("same_case_labels") is True
            and float(payload.get("endpoint_label_coverage", 0.0) or 0.0) > 0.0,
            {key: payload.get(key) for key in (
                "label_source", "same_case_labels", "endpoint_label_coverage",
            )},
        ))
    return rows


def _attack_sequences(path: Path) -> tuple[bool, dict]:
    if not path.is_file():
        return False, {"path": str(path), "reason": "missing_verified_jsonl"}
    count, seen, records = 0, set(), []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("source_id") or "")
            techniques = row.get("techniques")
            valid = (
                source_id and source_id not in seen and row.get("source_record")
                and re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_hash", "")).lower())
                and row.get("source_corpus")
                and row.get("source_corpus") == "AttackSeqBench"
                and isinstance(techniques, list) and techniques
                and all(TECHNIQUE_RE.fullmatch(str(value).upper()) for value in techniques)
            )
            if not valid:
                return False, {"path": str(path), "invalid_line": number}
            seen.add(source_id)
            records.append(row)
            count += 1
    except (OSError, json.JSONDecodeError):
        return False, {"path": str(path), "reason": "invalid_jsonl"}
    manifest_path = path.with_name("content_manifest.json")
    manifest = _read_json(manifest_path)
    from scripts.import_attack_sequence_records import _content_manifest
    expected_manifest = _content_manifest(records)
    aggregate = expected_manifest["aggregate_sha256"]
    manifest_ok = (
        isinstance(manifest, dict)
        and manifest.get("source_corpus") == "AttackSeqBench"
        and manifest.get("record_count") == count
        and manifest.get("records") == expected_manifest["records"]
        and manifest.get("aggregate_sha256") == aggregate
    )
    return count == 408 and manifest_ok, {
        "path": str(path), "records": count,
        "content_manifest": str(manifest_path),
        "aggregate_sha256": aggregate,
        "manifest_matches": manifest_ok,
    }


def _augmentation(path: Path):
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return [("augmentation run manifest", False, {"path": str(path), "reason": "missing_or_invalid"})]
    split = payload.get("split_contract") if isinstance(payload.get("split_contract"), dict) else {}
    train, test = set(split.get("train_snapshots", [])), set(split.get("test_snapshots", []))
    calls = payload.get("llm_calls") if isinstance(payload.get("llm_calls"), list) else []
    fields = {"stage", "attempt", "model", "provider", "status", "wall_latency_seconds"}
    return [
        ("augmentation train-only disjoint split", split.get("donor_policy") == "train_only" and bool(train) and bool(test) and train.isdisjoint(test), split),
        ("augmentation admitted variants", int(payload.get("admitted_count", 0) or 0) > 0, {"admitted_count": payload.get("admitted_count")}),
        ("augmentation LLM call telemetry", bool(calls) and all(isinstance(row, dict) and fields.issubset(row) for row in calls), {"calls": len(calls)}),
    ]


def _human_ratings(path: Path) -> tuple[bool, dict]:
    required = {
        "Variant_ID", "Model", "Condition", "Strategy", "Description",
        "R1", "R2", "R3", "Mean", "UsabilityFlag",
    }
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields_ok = required.issubset(reader.fieldnames or [])
    except OSError:
        return False, {"path": str(path), "reason": "missing_or_unreadable"}

    ids = [str(row.get("Variant_ID") or "") for row in rows]
    numeric_ok = True
    matrix: list[list[float]] = []
    for row in rows:
        try:
            scores = [float(row[key]) for key in ("R1", "R2", "R3")]
            stored_mean = float(row["Mean"])
        except (KeyError, TypeError, ValueError):
            numeric_ok = False
            continue
        mean = sum(scores) / 3.0
        numeric_ok = numeric_ok and all(value.is_integer() and 1 <= value <= 5 for value in scores)
        numeric_ok = numeric_ok and abs(stored_mean - round(mean, 2)) <= 0.011
        numeric_ok = numeric_ok and row.get("UsabilityFlag") == ("Y" if mean >= 4.0 else "N")
        matrix.append(scores)

    pair_counts = Counter((row.get("Model"), row.get("Condition")) for row in rows)
    models = {key[0] for key in pair_counts}
    conditions = {key[1] for key in pair_counts}
    balanced = (
        len(models) == 5 and len(conditions) == 6 and len(pair_counts) == 30
        and all(count == 100 for count in pair_counts.values())
    )
    try:
        from scripts.compute_rating_agreement import krippendorff_alpha_interval
        alpha = krippendorff_alpha_interval(matrix)
    except (ImportError, ValueError, ZeroDivisionError):
        alpha = float("nan")
    ok = (
        fields_ok and len(rows) == 3000 and all(ids) and len(ids) == len(set(ids))
        and numeric_ok and len(matrix) == len(rows) and balanced and abs(alpha - 0.83) <= 0.0001
    )
    return ok, {
        "path": str(path), "rows": len(rows), "raters": 3,
        "model_condition_cells": len(pair_counts), "rows_per_cell": sorted(set(pair_counts.values())),
        "overall_alpha_interval": round(alpha, 4) if alpha == alpha else None,
        "sha256": _sha256(path) if path.is_file() else None,
    }


def validate(runtime_dir: Path | None, augmentation_manifest: Path | None, attack_path: Path,
             ratings_path: Path, mmd_manifest: Path) -> dict:
    checks = []
    for rel in (
        "prompts/edge_mutation.txt", "prompts/replacement.txt", "prompts/rewriting.txt",
        "prompts/extension.txt", "scripts/run_augmentation.py", "scripts/run_detection.py",
        "scripts/run_interpretation.py", "src/utils/split.py",
    ):
        path = REPO_ROOT / rel
        checks.append((rel, path.is_file() and path.stat().st_size > 0, {"path": str(path)}))
    for dataset in DATASETS:
        labels = _label_files(dataset)
        checks.append((f"{dataset} nonempty malicious labels", bool(labels), {"files": len(labels)}))
        ok, count = _technique_schema(dataset)
        checks.append((f"{dataset} technique/tactic annotation schema", ok, {"valid_files": count}))
        if runtime_dir is not None:
            checks.extend(_runtime_checks(runtime_dir / f"{dataset}.json", dataset))
    for dataset in ("darpa_e5", "optc"):
        ok, detail = _pidsmaker_registry(dataset)
        checks.append((f"{dataset} PIDSMaker malicious-node provenance", ok, detail))
    ok, detail = _atlas_source_linked_registry()
    checks.append(("ATLAS v1 source-linked endpoint registry", ok, detail))
    required = {"host_0051", "host_0201", "host_0501"}
    found = {path.stem for path in _label_files("optc")}
    checks.append(("OpTC paper-host labels H051/H201/H501", required.issubset(found), {
        "missing": sorted(required - found),
    }))
    ok, detail = _attack_sequences(attack_path)
    checks.append(("AttackSeq records with source locator/hash/corpus", ok, detail))
    if augmentation_manifest is not None:
        checks.extend(_augmentation(augmentation_manifest))
    ok, detail = _human_ratings(ratings_path)
    checks.append(("human-rating sheet integrity and agreement", ok, detail))
    mmd = _read_json(mmd_manifest)
    checks.append(("MMD real-input permutation manifest", isinstance(mmd, dict)
                   and bool(mmd.get("reference_features_sha256"))
                   and bool(mmd.get("variant_features_sha256"))
                   and int(mmd.get("permutations", 0) or 0) > 0
                   and mmd.get("seed") is not None, {"path": str(mmd_manifest)}))
    rows = [{"name": name, "passed": bool(ok), "detail": detail} for name, ok, detail in checks]
    return {"passed": all(row["passed"] for row in rows),
            "passed_checks": sum(row["passed"] for row in rows),
            "failed_checks": sum(not row["passed"] for row in rows), "checks": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report-dir", type=Path,
                        help="optional directory of run-derived dataset reports")
    parser.add_argument("--augmentation-manifest", type=Path,
                        help="optional manifest from a completed augmentation run")
    parser.add_argument("--attack-sequences", type=Path, default=REPO_ROOT / "data" / "attack_knowledge" / "attackseqbench" / "verified_sequences.jsonl")
    parser.add_argument("--ratings", type=Path, default=REPO_ROOT / "data" / "human_ratings.csv")
    parser.add_argument("--mmd-manifest", type=Path, default=REPO_ROOT / "outputs" / "mmd" / "manifest.json")
    args = parser.parse_args(argv)
    result = validate(args.runtime_report_dir, args.augmentation_manifest,
                      args.attack_sequences, args.ratings, args.mmd_manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
