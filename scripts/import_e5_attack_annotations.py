"""Import reviewed DARPA TC E5 entity labels with exact event provenance.

The input registry is produced from official CDM20 Avro shards, PIDSMaker
malicious-node anchors, and the TA5.1 E5 report.  This importer deliberately
does not infer a technique from a scene-wide label: every accepted row must
match one immutable reviewed event pin and have exactly one occurrence of the
selected event UUID in its pinned raw shard.

The generated repository bundle is portable.  Local input paths are removed;
official shard names/hashes, raw record ordinals/hashes, PIDSMaker source rows,
and TA5.1 report locators remain available for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PIDSMaker_REPOSITORY = "https://github.com/ubc-provenance/PIDSMaker.git"
PIDSMaker_COMMIT = "32602734bc9f896be5fc0f03f0a185c967cd6624"
TA51_REPORT_PATH = "Ground_Truth/TA51_Final_report_E5.pdf"
TA51_REPORT_SHA256 = "aa12f7b93399159491f90fe036a88f1250c451809f686950e55edc0ee6a25d3b"
TAXONOMY_SHA256 = "6658c7d3fb66b72aa19c6377c902de4a29dcfd77e3e942d4996a0a21f3e9f482"
EXPECTED_REGISTRY_SHA256 = "ee598bfe190aaf34c4162be87ba9909439e72f52c08a445d1c434651b6b4417f"

# Canonical hashes of the six reviewed, path-sanitized input records.  These
# pins bind the event, anchor/role, raw shard/hash/ordinal, report locator,
# ATT&CK judgment, and full-shard occurrence audit as one unit.
REVIEWED_INPUT_SHA256 = {
    "e5-cadets-0517-c2-http-001": "5bbf73d4fc01e4ea9b80e130aff21eb9190076acc527cd3561d63c66b1d0025e",
    "e5-cadets-0517-nginx-exploit-001": "ff96c2e49d98317e5d94848c7f5788f85a3da0139ba8c3b943ef47f61cab618d",
    "e5-cadets-0517-stage-transfer-001": "52da730510a304805cacb2ca974c7d71d74f05bb4a8f55deff9d68c88a97e086",
    "e5-clearscope-0517-custom-c2-001": "85b18eca6a69d3feddb6191776d3896b3758bf004b32426f4d4e6ce7db583ce5",
    "e5-theia-0515-c2-http-001": "55e65ebf703943d69be4c25b89be00e65cd00082a4cebeb4bc527aba738534de",
    "e5-trace-0514-c2-http-001": "2e08234af43af05c4b4b0b7baf875a5a8d17dc417755cb6b5cfa63574ced533d",
}

EXPECTED_SHARDS = {
    "ta1-cadets-1-e5-official-2.bin.116.gz": "cf2a45b7d4db0d3f2efb2e691ec6693aa2a1666506211df0b13818b995d08933",
    "ta1-clearscope-1-e5-official-1.bin.33.gz": "386d46a9b525a65f953ee13fc30446cbef718cdbf6344b72e3b68837a601c56b",
    "ta1-theia-1-e5-official-2.bin.30.gz": "f39a4dca3730309c9b390b3f71cd608baf7133333d1e5e867ea24669088e452c",
    "ta1-trace-2-e5-official-1.bin.129.gz": "5ab20801c7806c053ed84d82c96e5967e3981354995ec80530412adf856e7d8c",
}

EXPECTED_TECHNIQUES = {
    "e5-cadets-0517-c2-http-001": ("T1071.001", "Command and Control"),
    "e5-cadets-0517-nginx-exploit-001": ("T1190", "Initial Access"),
    "e5-cadets-0517-stage-transfer-001": ("T1105", "Command and Control"),
    "e5-clearscope-0517-custom-c2-001": ("T1095", "Command and Control"),
    "e5-theia-0515-c2-http-001": ("T1071.001", "Command and Control"),
    "e5-trace-0514-c2-http-001": ("T1071.001", "Command and Control"),
}

EXPECTED_OCCURRENCES = {
    "e5-cadets-0517-c2-http-001": 574,
    "e5-cadets-0517-nginx-exploit-001": 3,
    "e5-cadets-0517-stage-transfer-001": 149,
    "e5-clearscope-0517-custom-c2-001": 67,
    "e5-theia-0515-c2-http-001": 55,
    "e5-trace-0514-c2-http-001": 39,
}

SCENE_AUDIT = [
    {"platform": "theia5", "attack_id": "theia_firefox_drakon_0515", "status": "exact", "exact_event_annotation_count": 1},
    {"platform": "cadets5", "attack_id": "cadets_nginx_drakon_0516", "status": "incomplete", "exact_event_annotation_count": 0,
     "reason": "The report and available shard evidence do not uniquely distinguish every stage/C2 flow on both hosts."},
    {"platform": "cadets5", "attack_id": "cadets_nginx_drakon_0517", "status": "exact", "exact_event_annotation_count": 3},
    {"platform": "trace5", "attack_id": "trace_firefox_drakon_0514", "status": "exact", "exact_event_annotation_count": 1},
    {"platform": "clearscope5", "attack_id": "clearscope_appstarter_0515", "status": "incomplete", "exact_event_annotation_count": 0,
     "reason": "No unique event/anchor/report join is released."},
    {"platform": "clearscope5", "attack_id": "clearscope_firefox_0517", "status": "incomplete", "exact_event_annotation_count": 0,
     "reason": "No unique event/anchor/report join is released."},
    {"platform": "clearscope5", "attack_id": "clearscope_lockwatch_0517", "status": "incomplete", "exact_event_annotation_count": 0,
     "reason": "No unique event/anchor/report join is released."},
    {"platform": "clearscope5", "attack_id": "clearscope_tester_0517", "status": "exact", "exact_event_annotation_count": 1},
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _portable_anchor_path(value) -> str:
    text = str(value or "")
    if text.startswith("Ground_Truth/"):
        return text
    marker = "/Ground_Truth/"
    if marker not in text:
        raise ValueError("E5 anchor source is not rooted at Ground_Truth")
    return "Ground_Truth/" + text.split(marker, 1)[1]


def _portable_input(row: dict) -> dict:
    keys = (
        "annotation_id", "status", "evidence_scope", "technique_assignment",
        "dataset", "platform", "attack_id", "host_id", "event_uuid",
        "timestamp_nanos", "timestamp_utc", "event_type", "actor_uuid",
        "object_uuid", "object2_uuid", "anchor_uuid", "anchor_role",
        "anchor_node_type", "anchor_attributes_raw", "report",
        "attack_technique", "raw_anchor_occurrence_audit",
    )
    result = {key: row.get(key) for key in keys}
    source = row.get("source") if isinstance(row.get("source"), dict) else {}
    result["source"] = {
        key: source.get(key) for key in (
            "producer", "official_avro_filename", "official_drive_file_id",
            "official_avro_sha256", "source_record_ordinal",
            "raw_event_json_sha256", "anchor_csv_sha256",
        )
    }
    result["source"]["anchor_csv"] = _portable_anchor_path(source.get("anchor_csv"))
    return result


def _load_reviewed(registry_path: Path, manifest_path: Path, tactic_path: Path) -> tuple[list[dict], dict, dict]:
    if _sha256(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise ValueError("E5 exact-event registry hash differs from the reviewed release")
    rows = [
        json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tactic_map = json.loads(tactic_path.read_text(encoding="utf-8"))
    if _sha256(tactic_path) != TAXONOMY_SHA256:
        raise ValueError("ATT&CK taxonomy file hash differs from the reviewed release")
    registry_meta = manifest.get("registry") if isinstance(manifest.get("registry"), dict) else {}
    report_meta = manifest.get("official_report") if isinstance(manifest.get("official_report"), dict) else {}
    if (
        registry_meta.get("sha256") != EXPECTED_REGISTRY_SHA256
        or int(registry_meta.get("rows", -1)) != 6
        or report_meta.get("sha256") != TA51_REPORT_SHA256
    ):
        raise ValueError("E5 source manifest does not bind the reviewed registry/report")
    by_id = {str(row.get("annotation_id") or ""): row for row in rows}
    if set(by_id) != set(REVIEWED_INPUT_SHA256) or len(by_id) != len(rows):
        raise ValueError("E5 registry annotation IDs differ from the reviewed six-event set")
    for annotation_id, row in by_id.items():
        portable = _portable_input(row)
        if _canonical_hash(portable) != REVIEWED_INPUT_SHA256[annotation_id]:
            raise ValueError(f"reviewed E5 event pin changed: {annotation_id}")
        source = portable["source"]
        report = portable["report"] if isinstance(portable.get("report"), dict) else {}
        attack = portable["attack_technique"] if isinstance(portable.get("attack_technique"), dict) else {}
        occurrence = portable["raw_anchor_occurrence_audit"] if isinstance(portable.get("raw_anchor_occurrence_audit"), dict) else {}
        expected_technique, expected_tactic = EXPECTED_TECHNIQUES[annotation_id]
        if (
            row.get("status") != "exact"
            or row.get("evidence_scope") != "event_and_report"
            or row.get("technique_assignment") != "final_event_gt"
            or row.get("snapshot") is not None
            or source.get("official_avro_sha256") != EXPECTED_SHARDS.get(source.get("official_avro_filename"))
            or report.get("sha256") != TA51_REPORT_SHA256
            or attack.get("technique_id") != expected_technique
            or attack.get("tactic") != expected_tactic
            or expected_tactic not in tactic_map.get(expected_technique, [])
            or occurrence.get("selected_event_matches") != 1
            or occurrence.get("events_containing_anchor") != EXPECTED_OCCURRENCES[annotation_id]
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("raw_event_json_sha256") or ""))
        ):
            raise ValueError(f"E5 event/report/taxonomy contract failed: {annotation_id}")
    return sorted(rows, key=lambda row: (int(row["timestamp_nanos"]), row["annotation_id"])), manifest, tactic_map


def _source_record(row: dict) -> dict:
    portable = _portable_input(row)
    source = portable["source"]
    report = portable["report"]
    attack = portable["attack_technique"]
    role = str(portable["anchor_role"])
    selected_object = portable["object2_uuid"] if role == "object2" else portable["object_uuid"]
    graph_event_id = portable["event_uuid"] + (
        ":predicateObject2" if role == "object2" else ":predicateObject"
    )
    evidence = {
        "record_type": "darpa_e5_manual_annotation_evidence",
        "schema": "athena.darpa_e5.attack_annotation_evidence.v1",
        "annotation_id": portable["annotation_id"],
        "dataset": portable["platform"],
        "source_attack_id": portable["attack_id"],
        "official_avro": {
            "filename": source["official_avro_filename"],
            "sha256": source["official_avro_sha256"],
            "drive_file_id": source["official_drive_file_id"],
            "record_ordinal": source["source_record_ordinal"],
            "producer": source["producer"],
        },
        "raw_event": {
            "raw_event_uuid": portable["event_uuid"],
            "graph_event_id": graph_event_id,
            "event_id": graph_event_id,
            "timestamp_nanos": portable["timestamp_nanos"],
            "timestamp": portable["timestamp_utc"],
            "host_id": portable["host_id"],
            "actorID": portable["actor_uuid"],
            "objectID": selected_object,
            "primaryObjectID": portable["object_uuid"],
            "object2ID": portable["object2_uuid"],
            "action": portable["event_type"],
            "source_record_sha256": source["raw_event_json_sha256"],
        },
        "anchor": {
            "role": role,
            "node_uuid": portable["anchor_uuid"],
            "node_type": portable["anchor_node_type"],
            "attributes": portable["anchor_attributes_raw"],
            "source_path": source["anchor_csv"],
            "source_file_sha256": source["anchor_csv_sha256"],
        },
        "official_report": {
            "source_repository": PIDSMaker_REPOSITORY,
            "source_commit": PIDSMaker_COMMIT,
            "path": TA51_REPORT_PATH,
            "sha256": report["sha256"],
            "section": report["section"],
            "pdf_pages": report["pdf_pages"],
            "pdftotext_line_start": report["pdftotext_line_start"],
            "pdftotext_line_end": report["pdftotext_line_end"],
            "evidence": report["evidence"],
        },
        "attack_taxonomy": {
            "technique": attack["technique_id"],
            "technique_name": attack["technique_name"],
            "tactic": attack["tactic"],
            "taxonomy_sha256": TAXONOMY_SHA256,
        },
        "review": {
            "status": "final_high_confidence",
            "evidence_scope": "event_and_report",
            "mapping_basis": attack["mapping_basis"],
            "selection_rule": portable["raw_anchor_occurrence_audit"]["selection_rule"],
        },
        "occurrence_audit": portable["raw_anchor_occurrence_audit"],
        "input_record_sha256": REVIEWED_INPUT_SHA256[portable["annotation_id"]],
    }
    evidence["record_sha256"] = _canonical_hash(evidence)
    return evidence


def build(registry_path: Path, manifest_path: Path, output_dir: Path, tactic_path: Path) -> dict:
    rows, input_manifest, tactic_map = _load_reviewed(registry_path, manifest_path, tactic_path)
    evidence_rows = [_source_record(row) for row in rows]
    evidence_rows.sort(key=lambda row: (row["raw_event"]["timestamp_nanos"], row["annotation_id"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "source_records.jsonl"
    evidence_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in evidence_rows),
        encoding="utf-8",
    )
    evidence_file_sha = _sha256(evidence_path)

    annotations = []
    evidence_by_id = {row["annotation_id"]: row for row in evidence_rows}
    for source in evidence_rows:
        raw = source["raw_event"]
        anchor = source["anchor"]
        taxonomy = source["attack_taxonomy"]
        annotations.append({
            "record_type": "source_linked_attack_annotation",
            "annotation_id": source["annotation_id"],
            "source_id": source["annotation_id"],
            "dataset": source["dataset"],
            "scene": "*",
            "scene_resolution": "runtime_unique_raw_event",
            "source_attack_id": source["source_attack_id"],
            "host_id": raw["host_id"],
            "anchor_uuid": anchor["node_uuid"],
            "anchor_role": anchor["role"],
            "event_id": raw["event_id"],
            "raw_event_uuid": raw["raw_event_uuid"],
            "event_timestamp": raw["timestamp"],
            "event_actor": raw["actorID"],
            "event_object": raw["objectID"],
            "event_action": raw["action"],
            "reference_technique": taxonomy["technique"],
            "reference_tactic": taxonomy["tactic"],
            "valid_tactics": tactic_map[taxonomy["technique"]],
            "annotation_status": "final_high_confidence",
            "annotation_protocol": "exact CDM20 event + PIDSMaker anchor + TA5.1 report action/time/endpoint join",
            "source_corpus": "DARPA TC E5 / PIDSMaker / TA5.1 final report",
            "source_locator": {
                "raw_event_id": raw["raw_event_uuid"],
                "graph_event_id": raw["graph_event_id"],
                "official_avro_filename": source["official_avro"]["filename"],
                "official_avro_sha256": source["official_avro"]["sha256"],
                "raw_event_json_sha256": raw["source_record_sha256"],
                "report_sha256": source["official_report"]["sha256"],
            },
            "source_record": "source_records.jsonl",
            "source_hash": evidence_file_sha,
            "source_record_sha256": source["record_sha256"],
        })
    annotations.sort(key=lambda row: (row["event_timestamp"], row["annotation_id"]))
    annotations_path = output_dir / "source_linked_annotations.jsonl"
    annotations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in annotations),
        encoding="utf-8",
    )

    mappings = []
    for annotation in annotations:
        source = evidence_by_id[annotation["annotation_id"]]
        raw = source["raw_event"]
        mappings.append({
            "record_type": "mapping",
            "dataset": annotation["dataset"],
            "scene": "*",
            "scene_resolution": annotation["scene_resolution"],
            "source_attack_id": annotation["source_attack_id"],
            "host_id": annotation["host_id"],
            "anchor_uuid": annotation["anchor_uuid"],
            "reference_technique": annotation["reference_technique"],
            "reference_tactic": annotation["reference_tactic"],
            "source_id": annotation["source_id"],
            "source_record": "source_records.jsonl",
            "source_hash": evidence_file_sha,
            "source_corpus": annotation["source_corpus"],
            "source_record_sha256": annotation["source_record_sha256"],
            "boundary": {
                "snapshot": None,
                "event_id": raw["event_id"],
                "raw_event_uuid": raw["raw_event_uuid"],
                "anchor": annotation["anchor_uuid"],
                "anchor_role": annotation["anchor_role"],
                "actor": raw["actorID"],
                "object": raw["objectID"],
                "action": raw["action"],
                "timestamp": raw["timestamp"],
                "timestamp_nanos": raw["timestamp_nanos"],
                "source_event_sha256": raw["source_record_sha256"],
            },
        })
    mapping_path = output_dir / "mapping_records.jsonl"
    mapping_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in mappings),
        encoding="utf-8",
    )

    audit_path = output_dir / "audit.md"
    audit_lines = [
        "# DARPA TC E5 entity-level ATT&CK labels with exact event provenance",
        "",
        "Only exact CDM event + PIDSMaker anchor + TA5.1 report joins are final.",
        "Incomplete attack scenes remain explicitly unannotated.",
        "",
        "| Dataset | Attack | Raw event UUID | Graph endpoint ID | Anchor / role | Raw anchor events | ATT&CK | Report |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for annotation in annotations:
        source = evidence_by_id[annotation["annotation_id"]]
        report = source["official_report"]
        audit_lines.append(
            f"| {annotation['dataset']} | {annotation['source_attack_id']} | `{annotation['raw_event_uuid']}` | "
            f"`{annotation['event_id']}` | "
            f"`{annotation['anchor_uuid']}` / {annotation['anchor_role']} | "
            f"{source['occurrence_audit']['events_containing_anchor']} (selected UUID=1) | "
            f"{annotation['reference_technique']} / {annotation['reference_tactic']} | "
            f"§{report['section']}, PDF p.{','.join(map(str, report['pdf_pages']))} |"
        )
    audit_lines.extend(["", "## Scene coverage", "", "| Platform | Attack | Status | Exact rows |", "|---|---|---:|---:|"])
    for scene in SCENE_AUDIT:
        audit_lines.append(
            f"| {scene['platform']} | {scene['attack_id']} | {scene['status']} | "
            f"{scene['exact_event_annotation_count']} |"
        )
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    source_generator = input_manifest.get("generator") if isinstance(input_manifest.get("generator"), dict) else {}
    occurrence_counter = input_manifest.get("raw_occurrence_counter") if isinstance(input_manifest.get("raw_occurrence_counter"), dict) else {}
    manifest = {
        "schema": "athena.darpa_e5.final_attack_annotations.v1",
        "dataset": "darpa_e5",
        "annotation_status": "final_high_confidence",
        "annotation_count": len(annotations),
        "exact_attack_scene_count": sum(scene["status"] == "exact" for scene in SCENE_AUDIT),
        "incomplete_attack_scene_count": sum(scene["status"] == "incomplete" for scene in SCENE_AUDIT),
        "paper_platforms": ["cadets5", "clearscope5", "theia5", "trace5"],
        "paper_profile_pidsmaker_attack_sources": 8,
        "five_directions_in_paper_profile": False,
        "coverage_policy": "six reviewed entity labels anchored to exact events; incomplete scenes are not assigned scene-wide techniques",
        "scene_audit": SCENE_AUDIT,
        "technique_counts": dict(sorted(Counter(row["reference_technique"] for row in annotations).items())),
        "official_report": {
            "repository": PIDSMaker_REPOSITORY,
            "commit": PIDSMaker_COMMIT,
            "path": TA51_REPORT_PATH,
            "sha256": TA51_REPORT_SHA256,
        },
        "official_shards": EXPECTED_SHARDS,
        "input_registry": {"sha256": EXPECTED_REGISTRY_SHA256, "rows": 6},
        "source_registry_generator": {"sha256": source_generator.get("sha256")},
        "raw_occurrence_counter": {"sha256": occurrence_counter.get("source_sha256")},
        "attack_taxonomy": {
            "path": "data/attack_knowledge/mitre_attack/technique_to_tactic.json",
            "sha256": TAXONOMY_SHA256,
        },
        "converter": {
            "path": "scripts/import_e5_attack_annotations.py",
            "sha256": _sha256(Path(__file__)),
        },
        "outputs": {
            "source_records.jsonl": _sha256(evidence_path),
            "source_linked_annotations.jsonl": _sha256(annotations_path),
            "mapping_records.jsonl": _sha256(mapping_path),
            "audit.md": _sha256(audit_path),
        },
    }
    manifest["aggregate_sha256"] = _canonical_hash(manifest)
    manifest_path_out = output_dir / "content_manifest.json"
    manifest_path_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-registry", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument(
        "--taxonomy", type=Path,
        default=REPO_ROOT / "data/attack_knowledge/mitre_attack/technique_to_tactic.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data/annotated_labels/darpa_e5/attack_techniques",
    )
    args = parser.parse_args(argv)
    manifest = build(args.input_registry, args.input_manifest, args.output_dir, args.taxonomy)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
