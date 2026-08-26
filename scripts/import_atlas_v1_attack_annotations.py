#!/usr/bin/env python3
"""Build reviewed ATT&CK mappings from official ATLAS-v1 endpoint joins.

The reviewed list is deliberately event-specific.  A scenario feature in
ATLAS Table 2 is necessary but never sufficient: the selected endpoint must
also expose the corresponding phishing URL/domain or downloaded attachment in
the official, same-case malicious endpoint registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.snapshot_construction.atlas_parser import (  # noqa: E402
    ATLASHandler, _preprocessed_case_labels,
)
from src.snapshot_construction.atlas_v1 import (  # noqa: E402
    convert_preprocessed_file, normalize_atlas_value, resolve_case_label_ids,
)
from src.utils.interval_replay import event_sha256  # noqa: E402

ATLAS_PAPER_URL = "https://www.usenix.org/system/files/sec21-alsaheel.pdf"
ATLAS_PAPER_SHA256 = "561e298c63cc1f30384eb1e07ee63911743f30214a5601c9f4a4eeb5fc5f199c"
ATLAS_REPOSITORY = "https://github.com/purseclab/ATLAS.git"
ATLAS_COMMIT = "e46096d1947e4f059e73a0ac2b9a9707812fd4bc"
TAXONOMY_SHA256 = "6658c7d3fb66b72aa19c6377c902de4a29dcfd77e3e942d4996a0a21f3e9f482"

# source_id: (scene, host, anchor, type, action, feature, technique, tactic)
#
# PL rows are restricted to the initial H1 malicious web object.  PA rows are
# restricted to the H1 downloaded document itself.  No INJ/IG/BD/LM endpoint
# is assigned a technique from the scenario-level feature alone.
REVIEWED = {
    "atlas-v1:bb10e56206c3e098676848fc6b476e6621582f8d1163cd432846add2b3870295":
        ("S1", "H1", "0xalsaheel.com:9999/utblfno", "web_object", "response", "PL", "T1566.002", "Initial Access"),
    "atlas-v1:ea7a444e4a76622396e4af6ae02d50eb523c696f7189aa43849da61998afc319":
        ("S2", "H1", "0xalsaheel.com:9999/ripleeszw", "web_object", "response", "PL", "T1566.002", "Initial Access"),
    "atlas-v1:3a044f65a2d106f28049777829f5e59bc9707c65692993f6d9092f1177d35261":
        ("M1", "H1", "0xalsaheel.com:9999/18f18fnc", "web_object", "response", "PL", "T1566.002", "Initial Access"),
    "atlas-v1:91e3c2190b6c8a5172fe50d58116eb873a447e1bb8020a8aecae482b863fdca3":
        ("M2", "H1", "0xalsaheel.com:9999/8ri2cz6pbm3", "web_object", "response", "PL", "T1566.002", "Initial Access"),
    "atlas-v1:363e4639cd664d9f45c33aa9aab9a7866b2102fad22f563c71537ef8d7c729f3":
        ("M3", "H1", "0xalsaheel.com:9999/18f18fnc", "web_object", "response", "PL", "T1566.002", "Initial Access"),
    "atlas-v1:a27c7523d6d26d7b70ae2c6100356c9e41e083eb3cc4195eaff927a25ea6818e":
        ("M5", "H1", "0xalsaheel.com/msf.doc", "web_object", "response", "PL", "T1566.002", "Initial Access"),
    "atlas-v1:2b44fa2ee647962b0da7a85291f7357e39f297545c203984be08838b33424eb4":
        ("S3", "H1", "c:/users/aalsahee/downloads/msf.rtf", "file", "file_read", "PA", "T1566.001", "Initial Access"),
    "atlas-v1:ef94c06db991c421b1468883c3735ed61041d8d650df97d21024f809b2e4591d":
        ("S4", "H1", "c:/users/aalsahee/downloads/msf.doc", "file", "file_read", "PA", "T1566.001", "Initial Access"),
    "atlas-v1:c07a05e82408d5c58ffa85c5ebe27bffc58421a95e11a16007f3c9cfe42efcb2":
        ("M4", "H1", "c:/users/aalsahee/downloads/msf_2018_8174.rtf", "file", "file_read", "PA", "T1566.001", "Initial Access"),
    "atlas-v1:a97d59d5dacbb7ab921ca22b6d4fc75da4eece554fe2c98bc5c81695e1571acb":
        ("M6", "H1", "c:/users/aalsahee/downloads/msf.rtf", "file", "file_read", "PA", "T1566.001", "Initial Access"),
}

# Immutable reviewed event pins.  For PL the selected event is the unique
# occurrence already stored in the endpoint registry.  For PA it is the first
# official-input-order Firefox file_write to the downloaded attachment after
# enumerating every occurrence in the canonical testing stream.
BOUNDARY_PINS = {
    "atlas-v1:bb10e56206c3e098676848fc6b476e6621582f8d1163cd432846add2b3870295":
        (34, 1261, 34, "atlas-v1:999f3f442376cb1f1e90f88b6e0e7c99667e5de1bcceeec3460447da9fbd0ff4", 4791, 26770886.0, "3971238821c8a4fb57ca5da3d94fa3fefbf80c19fcdf82931c5c5268fe653a68"),
    "atlas-v1:ea7a444e4a76622396e4af6ae02d50eb523c696f7189aa43849da61998afc319":
        (239, 1466, 194, "atlas-v1:5f719e852dd1d919adbc1e018cafd095af15814b03448176e19deae63d0b013c", 107, 22319930.0, "04d238bc2b148d49704407067ee406a004d462de75075c6b21fd8dce2164ce88"),
    "atlas-v1:3a044f65a2d106f28049777829f5e59bc9707c65692993f6d9092f1177d35261":
        (42, 42, 42, "atlas-v1:68ec497853b37071e14414eb160192967f7affa2fe16f8c0728eea3ca1836a35", 62, 30731642.0, "df339bae855e7924434a4bc87b76d8658f9df694214169bd15f2e1a78661f114"),
    "atlas-v1:91e3c2190b6c8a5172fe50d58116eb873a447e1bb8020a8aecae482b863fdca3":
        (209, 209, 46, "atlas-v1:47778d17a5d03b911ab686fe09edaa848207eb94d9c297fee7165759f29d4323", 3908, 30792220.0, "afe70050cff16863793c4382f9cc5318b9630d549a54f9c588eadddb21e486e5"),
    "atlas-v1:363e4639cd664d9f45c33aa9aab9a7866b2102fad22f563c71537ef8d7c729f3":
        (387, 387, 53, "atlas-v1:12ceec225722b563f01ac291084cc9024dfd7766660187581dd22569d547da2f", 5091, 30739550.0, "420724302c7409c69a442ab44fbe4c834f3e046312d781f377b73301f2b2e344"),
    "atlas-v1:a27c7523d6d26d7b70ae2c6100356c9e41e083eb3cc4195eaff927a25ea6818e":
        (848, 848, 177, "atlas-v1:8a189e38d63d470e69a74c6c01295a36ba98292faf55005e62164bebe05b243d", 4336, 30208283.0, "c5526287e597e65056a52afa8577e05fdb6ed5a6344b5335938fc624c222b31b"),
    "atlas-v1:2b44fa2ee647962b0da7a85291f7357e39f297545c203984be08838b33424eb4":
        (308, 1535, 27, "atlas-v1:1aee6aa5cd00a17f8c2f78f9503de354c01d4543d4174a693091ef8e8e78ccfb", 72, 28997160.0, "50521fa1d74292376c696c03dabf16a8ed2d47c3f69bbce5edd23db5cb37d01f"),
    "atlas-v1:ef94c06db991c421b1468883c3735ed61041d8d650df97d21024f809b2e4591d":
        (330, 1557, 12, "atlas-v1:8609a41546315dbcb550c9e7951ca245f6fd90b4606fde6f536042eb7c039304", 783, 29272335.0, "3cd26910ca9efd034a427726aa89a3381986532eb39c8525d4085c2fa774bc5f"),
    "atlas-v1:c07a05e82408d5c58ffa85c5ebe27bffc58421a95e11a16007f3c9cfe42efcb2":
        (561, 561, 35, "atlas-v1:77eb184c27195d7ccd10fbe4635508c4fd88d4a0af53ce7d7b9ada927e31c396", 2731, 30758470.0, "1f17cb2c565a3a4f56f9b53b61532db92523157c7a8f5d460a3bccf92da1def8"),
    "atlas-v1:a97d59d5dacbb7ab921ca22b6d4fc75da4eece554fe2c98bc5c81695e1571acb":
        (1042, 1042, 48, "atlas-v1:0166cb72bc210f60ca113d0c55e6549f5c0cde76d35a3f9cc1458e7fc6babed8", 2986, 30744615.0, "d0671ba930a84f9fb71c42bd1c9dbf94fff385d12b3c322e4482d6283740b2ab"),
}
PA_OCCURRENCE_COUNTS = {"S3": (96, 3), "S4": (133, 3), "M4": (82, 4), "M6": (79, 3)}
AUDIT_GLOBAL_FAMILY_BASE = {"M": 0, "S": 1227}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ), encoding="utf-8")


def _boundary_pin(boundary: dict) -> tuple:
    return (
        int(boundary.get("snapshot", -1)), int(boundary.get("source_global_snapshot", -1)),
        int(boundary.get("snapshot_local", -1)),
        str(boundary.get("event_id") or ""), int(boundary.get("event_order", -1)),
        float(boundary.get("event_timestamp", 0.0)), str(boundary.get("event_sha256") or ""),
    )


def _family_boundary(scene: str, boundary: dict) -> dict:
    """Separate ten-scene audit coordinates from family-local runtime IDs."""
    value = dict(boundary)
    global_snapshot = int(value.get("source_global_snapshot", value.get("snapshot", -1)))
    base = AUDIT_GLOBAL_FAMILY_BASE[str(scene)[0]]
    runtime_snapshot = global_snapshot - base
    if runtime_snapshot < 0:
        raise ValueError(f"invalid {scene} audit-global snapshot {global_snapshot}")
    value["source_global_snapshot"] = global_snapshot
    value["snapshot"] = runtime_snapshot
    return value


def _pa_occurrence_audit(input_root: Path, endpoint: dict) -> dict:
    """Enumerate one attachment's complete official testing-stream occurrence set."""
    scene, host = str(endpoint["scene"]), str(endpoint["host"])
    anchor = normalize_atlas_value(endpoint["resolved_anchor"])
    source = endpoint["source"]
    path = input_root / str(source["preprocessed_path"])
    if not path.is_file() or _sha256(path) != str(source["preprocessed_sha256"]):
        raise ValueError(f"canonical ATLAS stream is absent or changed for {scene}/{host}")
    frame = convert_preprocessed_file(path)
    case_dir, labels = _preprocessed_case_labels(input_root, path)
    resolved, _coverage = resolve_case_label_ids(case_dir, labels, frame)
    frame["_atlas_scenario"], frame["_atlas_host"] = scene, host
    handler = ATLASHandler(input_root, True, scene_name=scene)
    handler.scenario_labels[scene] = resolved
    graphs = handler.create_snapshots_from_graph(frame, True)
    source_boundary = endpoint["boundary"]
    global_offset = int(source_boundary["snapshot"]) - int(source_boundary["snapshot_local"])
    occurrences = []
    for local_snapshot, graph in enumerate(graphs):
        for edge_index, edge in enumerate(graph.es):
            source_anchor = normalize_atlas_value(graph.vs[edge.source]["name"])
            target_anchor = normalize_atlas_value(graph.vs[edge.target]["name"])
            if anchor not in {source_anchor, target_anchor}:
                continue
            role = "source" if source_anchor == anchor else "target"
            peer_vertex = graph.vs[edge.target if role == "source" else edge.source]
            attrs = edge.attributes()
            occurrences.append(_family_boundary(scene, {
                "snapshot": global_offset + local_snapshot,
                "snapshot_local": local_snapshot,
                "event_id": str(attrs.get("event_id") or ""),
                "event_order": int(attrs.get("event_order", -1)),
                "event_timestamp": float(attrs.get("timestamp", 0.0)),
                "action": str(attrs.get("actions") or ""),
                "endpoint_role": role, "anchor": anchor,
                "anchor_type": str(graph.vs[edge.source if role == "source" else edge.target]["type"]),
                "peer": normalize_atlas_value(peer_vertex["name"]),
                "peer_type": str(peer_vertex["type"]),
                "event_sha256": event_sha256(graph, edge_index),
            }))
    occurrences.sort(key=lambda row: (
        row["event_timestamp"], row["event_order"], row["event_id"], row["endpoint_role"],
    ))
    delivery = [
        row for row in occurrences
        if row["action"] == "file_write"
        and row["endpoint_role"] == "target"
        and row["peer_type"] == "process"
        and "firefox.exe_" in row["peer"]
    ]
    expected_total, expected_delivery = PA_OCCURRENCE_COUNTS[scene]
    if len(occurrences) != expected_total or len(delivery) != expected_delivery:
        raise ValueError(
            f"PA occurrence population changed for {scene}: "
            f"{len(occurrences)}/{len(delivery)}"
        )
    selected = delivery[0]
    source_id = str(endpoint["source_id"])
    if _boundary_pin(selected) != BOUNDARY_PINS[source_id]:
        raise ValueError(f"reviewed PA delivery boundary changed for {scene}")
    return {
        "schema": "athena.atlas_v1.attachment_occurrence_audit.v1",
        "source_preprocessed_path": str(source["preprocessed_path"]),
        "source_preprocessed_sha256": str(source["preprocessed_sha256"]),
        "selection_rule": (
            "enumerate every edge incident to the exact attachment endpoint; retain Firefox "
            "process->attachment file_write delivery events; select the earliest by "
            "(timestamp, official event_order, event_id)"
        ),
        "occurrence_count": len(occurrences),
        "delivery_candidate_count": len(delivery),
        "occurrences": occurrences,
        "delivery_candidates": delivery,
        "selected_boundary": selected,
    }


def _judgment(feature: str, endpoint: dict) -> dict:
    boundary = endpoint["boundary"]
    if feature == "PL":
        return {
            "criterion": "table2_pl_plus_initial_host_malicious_web_object",
            "basis": (
                "ATLAS Table 2 marks this scenario PL (phishing email link), and the selected "
                "official H1 event is a response from a web-object endpoint containing the "
                "same-case malicious 0xalsaheel.com label."
            ),
            "endpoint_evidence": {
                "required_type": "web_object", "required_action": "response",
                "required_label": "0xalsaheel.com",
            },
        }
    suffix = Path(str(boundary["anchor"])).suffix.lower()
    return {
        "criterion": "table2_pa_plus_first_browser_delivery_write",
        "basis": (
            "ATLAS Table 2 marks this scenario PA (phishing email attachment).  Across every "
            "official occurrence of the exact H1 attachment endpoint, the selected event is "
            "the first official-input-order Firefox process file_write delivering that file."
        ),
        "endpoint_evidence": {
            "required_type": "file", "required_action": "file_write",
            "required_role": "target", "required_peer": "firefox.exe process",
            "required_path_prefix": "c:/users/aalsahee/downloads/msf",
            "required_document_suffix": suffix,
        },
    }


def build(source_registry: Path, paper_pdf: Path, input_root: Path, output_dir: Path) -> dict:
    if _sha256(paper_pdf) != ATLAS_PAPER_SHA256:
        raise ValueError("paper PDF does not match the pinned official ATLAS USENIX 2021 paper")
    source_rows = [
        json.loads(line) for line in source_registry.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row.get("source_id") or ""): row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("ATLAS endpoint registry contains duplicate source IDs")
    source_registry_sha = _sha256(source_registry)
    taxonomy_path = REPO_ROOT / "data/attack_knowledge/mitre_attack/technique_to_tactic.json"
    if _sha256(taxonomy_path) != TAXONOMY_SHA256:
        raise ValueError("ATT&CK taxonomy registry differs from the reviewed version")
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    evidence_rows, drafts = [], []
    for source_id, reviewed in REVIEWED.items():
        scene, host, anchor, anchor_type, action, feature, technique, tactic = reviewed
        endpoint = by_id.get(source_id)
        if endpoint is None:
            raise ValueError(f"reviewed endpoint is absent: {source_id}")
        boundary = endpoint.get("boundary") or {}
        actual = (
            endpoint.get("scene"), endpoint.get("host"), boundary.get("anchor"),
            boundary.get("anchor_type"), boundary.get("action"),
        )
        if actual != (scene, host, anchor, anchor_type, action):
            raise ValueError(f"reviewed endpoint changed: {source_id}")
        if feature not in (endpoint.get("paper_attack_features") or []):
            raise ValueError(f"ATLAS Table 2 feature {feature} is absent for {scene}")
        if host != "H1":
            raise ValueError("phishing mappings must be tied to the initial H1 endpoint")
        if feature == "PL" and (
            "0xalsaheel.com" not in anchor or "0xalsaheel.com" not in endpoint.get("matched_labels", [])
        ):
            raise ValueError(f"PL endpoint is not tied to the official malicious domain: {source_id}")
        if feature == "PA" and (
            not anchor.startswith("c:/users/aalsahee/downloads/msf")
            or Path(anchor).suffix.lower() not in {".doc", ".rtf"}
            or not any(label in anchor for label in endpoint.get("matched_labels", []))
        ):
            raise ValueError(f"PA endpoint is not the official downloaded attachment: {source_id}")
        if tactic not in taxonomy.get(technique, []):
            raise ValueError(f"taxonomy does not map {technique} to {tactic}")
        if feature == "PL":
            if int(endpoint.get("matched_event_occurrence_count", -1)) != 1:
                raise ValueError(f"PL endpoint is not a unique official occurrence: {source_id}")
            selected_boundary = _family_boundary(scene, boundary)
            if _boundary_pin(selected_boundary) != BOUNDARY_PINS[source_id]:
                raise ValueError(f"reviewed PL boundary changed: {source_id}")
            occurrence_audit = {
                "schema": "athena.atlas_v1.unique_endpoint_occurrence.v1",
                "occurrence_count": 1, "selected_boundary": selected_boundary,
                "selection_rule": "the official malicious web-object endpoint has one occurrence",
            }
        else:
            occurrence_audit = _pa_occurrence_audit(input_root, endpoint)
            selected_boundary = occurrence_audit["selected_boundary"]
        judgment = _judgment(feature, endpoint)
        evidence = {
            "record_type": "atlas_v1_manual_annotation_evidence",
            "schema": "athena.atlas_v1.attack_annotation_evidence.v1",
            "endpoint_registry": {
                "path": "../source_linked/atlas_v1_source_link_endpoints.jsonl",
                "sha256": source_registry_sha,
                "source_id": source_id,
                "derived_sha256": endpoint["derived_sha256"],
            },
            "endpoint_record": endpoint,
            "occurrence_audit": occurrence_audit,
            "selected_boundary": selected_boundary,
            "paper_source": {
                "title": "ATLAS: A Sequence-based Learning Approach for Attack Investigation",
                "url": ATLAS_PAPER_URL, "sha256": ATLAS_PAPER_SHA256,
                "feature_reference": "Table 2, PDF page 12, proceedings page 3015",
                "feature": feature,
            },
            "review": judgment,
            "attack_taxonomy": {
                "path": "data/attack_knowledge/mitre_attack/technique_to_tactic.json",
                "sha256": TAXONOMY_SHA256,
                "technique": technique, "tactic": tactic,
            },
        }
        evidence["record_sha256"] = _canonical_hash(evidence)
        evidence_rows.append(evidence)
        drafts.append((endpoint, evidence, selected_boundary, technique, tactic, judgment))

    evidence_rows.sort(key=lambda row: (
        row["endpoint_record"]["scene"], row["endpoint_record"]["host"],
        row["endpoint_record"]["boundary"]["event_timestamp"],
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "source_records.jsonl"
    _write_jsonl(evidence_path, evidence_rows)
    evidence_file_sha = _sha256(evidence_path)
    evidence_by_endpoint = {
        row["endpoint_record"]["source_id"]: row for row in evidence_rows
    }

    annotations, mappings = [], []
    for endpoint, _evidence, boundary, technique, tactic, judgment in drafts:
        evidence = evidence_by_endpoint[endpoint["source_id"]]
        annotation_id = "atlas-attck:" + hashlib.sha256(
            f"{endpoint['source_id']}:{technique}".encode("utf-8")
        ).hexdigest()
        annotation = {
            "record_type": "source_linked_attack_annotation",
            "annotation_status": "final_high_confidence",
            "annotation_id": annotation_id, "source_id": annotation_id,
            "dataset": "atlas", "scene": endpoint["scene"], "host_id": endpoint["host"],
            "snapshot": int(boundary["snapshot"]), "event_id": boundary["event_id"],
            "source_global_snapshot": int(boundary["source_global_snapshot"]),
            "event_action": boundary["action"], "event_timestamp": boundary["event_timestamp"],
            "source_event_sha256": boundary["event_sha256"],
            "anchor_uuid": boundary["anchor"], "anchor_role": boundary["endpoint_role"],
            "anchor_type": boundary["anchor_type"],
            "reference_technique": technique, "reference_tactic": tactic,
            "valid_tactics": taxonomy[technique],
            "annotation_protocol": "ATLAS Table 2 feature + exact official-v1 malicious endpoint event + endpoint-specific manual review",
            "judgment_basis": judgment,
            "source_corpus": "official purseclab ATLAS v1 + USENIX Security 2021 paper",
            "source_record": "source_records.jsonl", "source_hash": evidence_file_sha,
            "source_record_sha256": evidence["record_sha256"],
            "source_locator": {
                "endpoint_source_id": endpoint["source_id"],
                "raw_event_id": boundary["event_id"],
                "paper_sha256": ATLAS_PAPER_SHA256,
            },
        }
        annotations.append(annotation)
        mappings.append({
            "record_type": "mapping", "dataset": "atlas",
            "scene": endpoint["scene"], "host_id": endpoint["host"],
            "anchor_uuid": boundary["anchor"],
            "reference_technique": technique, "reference_tactic": tactic,
            "boundary": {
                "snapshot": int(boundary["snapshot"]), "event_id": boundary["event_id"],
                "source_global_snapshot": int(boundary["source_global_snapshot"]),
                "anchor": boundary["anchor"], "event_sha256": boundary["event_sha256"],
                "source_event_sha256": boundary["event_sha256"],
                "action": boundary["action"], "timestamp": boundary["event_timestamp"],
            },
            "source_corpus": annotation["source_corpus"], "source_id": annotation_id,
            "source_record": "source_records.jsonl", "source_hash": evidence_file_sha,
            "source_record_sha256": evidence["record_sha256"],
        })
    annotations.sort(key=lambda row: (row["scene"], row["event_timestamp"], row["anchor_uuid"]))
    mappings.sort(key=lambda row: (row["scene"], row["boundary"]["timestamp"], row["anchor_uuid"]))
    annotations_path = output_dir / "source_linked_annotations.jsonl"
    mappings_path = output_dir / "mapping_records.jsonl"
    _write_jsonl(annotations_path, annotations)
    _write_jsonl(mappings_path, mappings)

    manifest = {
        "schema": "athena.atlas_v1.final_attack_annotations.v1",
        "dataset": "atlas", "annotation_count": len(annotations),
        "annotation_status": "final_high_confidence",
        "source_repository": ATLAS_REPOSITORY, "source_commit": ATLAS_COMMIT,
        "endpoint_registry": {
            "path": "../source_linked/atlas_v1_source_link_endpoints.jsonl",
            "sha256": source_registry_sha, "record_count": len(source_rows),
        },
        "snapshot_coordinates": {
            "mapping_snapshot": "ATLAS runtime family-local index",
            "source_global_snapshot": "ten-scenario source-audit index",
            "audit_global_family_base": AUDIT_GLOBAL_FAMILY_BASE,
        },
        "paper_source": {
            "url": ATLAS_PAPER_URL, "sha256": ATLAS_PAPER_SHA256,
            "reference": "Table 2, PDF page 12, proceedings page 3015",
        },
        "attack_taxonomy": {
            "path": "data/attack_knowledge/mitre_attack/technique_to_tactic.json",
            "sha256": TAXONOMY_SHA256,
        },
        "technique_counts": dict(sorted(Counter(
            row["reference_technique"] for row in annotations
        ).items())),
        "review_scope": {
            "included": ["PL URL/web-object endpoints", "PA downloaded attachment endpoints"],
            "pa_occurrence_counts": {
                scene: {"all": counts[0], "browser_delivery_candidates": counts[1]}
                for scene, counts in sorted(PA_OCCURRENCE_COUNTS.items())
            },
            "not_assigned_from_scene_feature_alone": ["INJ", "IG", "BD", "LM", "DE"],
            "m5_t1041": "not assigned because the endpoint registry has no exact data-transfer event",
        },
        "outputs": {
            evidence_path.name: _sha256(evidence_path),
            annotations_path.name: _sha256(annotations_path),
            mappings_path.name: _sha256(mappings_path),
        },
        "converter": {"path": "scripts/import_atlas_v1_attack_annotations.py",
                      "sha256": _sha256(Path(__file__))},
    }
    manifest["aggregate_sha256"] = _canonical_hash(manifest)
    manifest_path = output_dir / "content_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return {"annotations": annotations_path, "mappings": mappings_path,
            "evidence": evidence_path, "manifest": manifest_path, "count": len(annotations)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-registry", type=Path,
        default=REPO_ROOT / "data/annotated_labels/atlas/source_linked/atlas_v1_source_link_endpoints.jsonl",
    )
    parser.add_argument("--paper-pdf", type=Path, required=True)
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="unpacked official paper_experiments root used to enumerate PA occurrences",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data/annotated_labels/atlas/attack_techniques",
    )
    args = parser.parse_args(argv)
    result = build(args.source_registry, args.paper_pdf, args.input_root, args.output_dir)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
