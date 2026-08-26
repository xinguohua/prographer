#!/usr/bin/env python3
"""Build source-linked endpoint evidence from official purseclab ATLAS v1.

The input root must contain unpacked ``paper_experiments`` scenario folders
S1--S4 and M1--M6.  Only each case's canonical testing stream and same-case
``malicious_labels.txt`` are consumed.  ATT&CK values in the output are
explicitly human mapping candidates, never official ATLAS technique labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.snapshot_construction.atlas_parser import (  # noqa: E402
    ATLASHandler, _preprocessed_case_labels, atlas_host_from_path,
    atlas_scenario_from_path,
)
from src.snapshot_construction.atlas_v1 import (  # noqa: E402
    convert_preprocessed_file, discover_preprocessed_files,
    normalize_atlas_value, resolve_case_label_ids,
)
from src.utils.interval_replay import event_sha256  # noqa: E402


SCENARIOS = ("S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4", "M5", "M6")
FEATURES = {
    "S1": ["PL", "INJ", "IG", "BD", "DE"],
    "S2": ["PL", "INJ", "IG", "BD", "DE"],
    "S3": ["PA", "INJ", "IG", "BD", "DE"],
    "S4": ["PA", "INJ", "IG", "BD", "DE"],
    "M1": ["PL", "INJ", "IG", "BD", "LM", "DE"],
    "M2": ["PL", "INJ", "IG", "BD", "LM", "DE"],
    "M3": ["PL", "INJ", "IG", "BD", "LM", "DE"],
    "M4": ["PA", "INJ", "IG", "BD", "LM", "DE"],
    "M5": ["PL", "INJ", "IG", "BD", "LM", "DE"],
    "M6": ["PA", "INJ", "IG", "BD", "LM", "DE"],
}
FEATURE_DEFINITIONS = {
    "PL": "Phishing email link", "PA": "Phishing email attachment",
    "INJ": "Injection", "IG": "information gathering", "BD": "backdoor",
    "LM": "Lateral movement", "DE": "Data ex-filtration",
}


def _canonical(value) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping_candidates(scenario: str) -> list[dict]:
    rows = []
    features = FEATURES[scenario]
    definitions = (
        ("PL", "T1566.002", "Spearphishing Link", "technique",
         "https://attack.mitre.org/techniques/T1566/002/"),
        ("PA", "T1566.001", "Spearphishing Attachment", "technique",
         "https://attack.mitre.org/techniques/T1566/001/"),
        ("LM", "TA0008", "Lateral Movement", "tactic",
         "https://attack.mitre.org/tactics/TA0008/"),
        ("DE", "TA0010", "Exfiltration", "tactic",
         "https://attack.mitre.org/tactics/TA0010/"),
    )
    for feature, attack_id, name, kind, source in definitions:
        if feature in features:
            rows.append({
                "id": attack_id, "name": name, "kind": kind,
                "confidence": "high",
                "status": "candidate_not_official_atlas_annotation",
                "basis": f"ATLAS Table 2 explicitly defines {feature} as "
                         f"{FEATURE_DEFINITIONS[feature]}; MITRE defines {attack_id} as {name}.",
                "mitre_source": source,
            })
    if scenario == "M5":
        rows.append({
            "id": "T1041", "name": "Exfiltration Over C2 Channel",
            "kind": "technique", "confidence": "high",
            "status": "candidate_not_official_atlas_annotation",
            "basis": "ATLAS Section 6.5 states that the M5 backdoor leaks a secret file to a C&C server; MITRE T1041 is exfiltration over an existing command-and-control channel.",
            "mitre_source": "https://attack.mitre.org/techniques/T1041/",
        })
    return rows


def _paper_evidence(paper: Path) -> dict:
    return {
        "title": "ATLAS: A Sequence-based Learning Approach for Attack Investigation",
        "url": "https://www.usenix.org/system/files/sec21-alsaheel.pdf",
        "sha256": _sha256(paper),
        "feature_evidence": "Table 2, PDF page 12, proceedings page 3015",
        "m5_c2_evidence": "Section 6.5, Figure 9 discussion, proceedings page 3017",
    }


def _attack_evidence(paper: Path) -> dict:
    return {
        "separation": "Exact event/anchor joins are official-data-derived. ATT&CK entries are human mapping candidates and are not ATLAS ground-truth labels.",
        "paper_source": _paper_evidence(paper),
        "scenarios": [{
            "scenario": scenario,
            "paper_features": FEATURES[scenario],
            "paper_feature_definitions": FEATURE_DEFINITIONS,
            "paper_evidence_status": "direct_table_evidence",
            "attack_mapping_candidates": _mapping_candidates(scenario),
            "unmapped_to_specific_technique": [
                value for value in FEATURES[scenario] if value in {"INJ", "IG", "BD"}
            ],
        } for scenario in SCENARIOS],
    }


def build(input_root: Path, zip_dir: Path, paper: Path, output_dir: Path,
          source_commit: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = {}
    for scenario in SCENARIOS:
        archive = zip_dir / f"{scenario}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"missing official ATLAS archive: {archive}")
        archives[scenario] = {
            "path": archive.name, "sha256": _sha256(archive),
            "size_bytes": archive.stat().st_size,
        }
    paths = [
        path for path in discover_preprocessed_files(input_root, SCENARIOS)
        if path.name.startswith("testing_preprocessed_logs_")
    ]
    if len(paths) != 16:
        raise RuntimeError(f"expected 16 canonical testing host streams, found {len(paths)}")
    audit = {
        "schema": "athena.atlas_v1.source_link_audit.v1",
        "source_repository": {
            "url": "https://github.com/purseclab/ATLAS.git",
            "commit": source_commit,
            "release": "ATLAS USENIX Security 2021 paper_experiments",
        },
        "parser": {
            "event_converter": "src.snapshot_construction.atlas_v1.convert_preprocessed_file",
            "label_projection": "official normalized endpoint substring",
            "snapshot_builder": "src.snapshot_construction.atlas_parser.ATLASHandler.create_snapshots_from_graph",
            "window_seconds": 60,
        },
        "archives": archives, "cases": [], "summary": {},
        "attack_candidates": [], "attack_evidence": _attack_evidence(paper),
    }
    totals = {"event_count": 0, "snapshot_count": 0, "node_occurrences": 0,
              "edge_count": 0, "matched_endpoint_join_count": 0}
    global_snapshot = 0
    for path in paths:
        scenario, host = atlas_scenario_from_path(path), atlas_host_from_path(path)
        frame = convert_preprocessed_file(path)
        case_dir, labels = _preprocessed_case_labels(input_root, path)
        resolved, coverage = resolve_case_label_ids(case_dir, labels, frame)
        coverage["label_source"] = str((case_dir / "malicious_labels.txt").relative_to(input_root))
        frame["_atlas_scenario"], frame["_atlas_host"] = scenario, host
        handler = ATLASHandler(input_root, True, scene_name=scenario)
        handler.scenario_labels[scenario] = resolved
        graphs = handler.create_snapshots_from_graph(frame, True)
        source_ref = f"{scenario}:{host}:{path.name}"
        case = {
            "source_ref": source_ref, "scenario": scenario, "host": host,
            "case": case_dir.name,
            "source": {
                "zip": f"{scenario}.zip", "zip_sha256": archives[scenario]["sha256"],
                "preprocessed_path": str(path.relative_to(input_root)),
                "preprocessed_sha256": _sha256(path),
                "malicious_labels_path": str((case_dir / "malicious_labels.txt").relative_to(input_root)),
                "malicious_labels_sha256": _sha256(case_dir / "malicious_labels.txt"),
            },
            "labels": sorted(labels), "resolved_endpoints": sorted(resolved),
            "coverage": coverage, "event_count": len(frame),
            "snapshot_count": len(graphs),
            "node_occurrences": sum(graph.vcount() for graph in graphs),
            "edge_count": sum(graph.ecount() for graph in graphs),
            "matched_endpoint_joins": [],
        }
        candidates, occurrences = {}, {}
        for local_snapshot, graph in enumerate(graphs):
            snapshot = global_snapshot
            global_snapshot += 1
            for edge_index, edge in enumerate(graph.es):
                attrs = edge.attributes()
                matched = json.loads(str(attrs.get("matched_labels", "[]") or "[]"))
                if not matched:
                    continue
                for role, vertex in (("source", graph.vs[edge.source]),
                                     ("target", graph.vs[edge.target])):
                    anchor = normalize_atlas_value(vertex["name"])
                    endpoint_labels = sorted(label for label in matched if label in anchor)
                    if not endpoint_labels:
                        continue
                    if int(vertex["label"]) != 1:
                        raise RuntimeError(f"matched endpoint is not graph-positive: {anchor}")
                    occurrences[anchor] = occurrences.get(anchor, 0) + 1
                    row = {
                        "source_ref": source_ref, "scenario": scenario, "host": host,
                        "snapshot": snapshot, "snapshot_local": local_snapshot,
                        "window_start": graph["window_start"],
                        "event_id": str(attrs.get("event_id", "")),
                        "event_order": int(attrs.get("event_order", -1)),
                        "event_timestamp": float(attrs.get("timestamp", 0.0)),
                        "action": str(attrs.get("actions", "")), "endpoint_role": role,
                        "anchor": anchor, "anchor_type": str(vertex["type"]),
                        "matched_labels": endpoint_labels, "_graph": graph,
                        "_edge_index": edge_index,
                    }
                    order = (row["event_timestamp"], row["event_order"], snapshot,
                             row["event_id"], role)
                    if anchor not in candidates or order < candidates[anchor][0]:
                        candidates[anchor] = (order, row)
        for anchor in sorted(candidates):
            row = candidates[anchor][1]
            graph, edge_index = row.pop("_graph"), row.pop("_edge_index")
            row["event_sha256"] = event_sha256(graph, edge_index)
            row["matched_event_occurrence_count"] = occurrences[anchor]
            case["matched_endpoint_joins"].append(row)
        case["matched_endpoint_join_count"] = len(case["matched_endpoint_joins"])
        if {row["anchor"] for row in case["matched_endpoint_joins"]} != set(resolved):
            raise RuntimeError(f"{scenario}/{host} endpoint join set is incomplete")
        audit["cases"].append(case)
        for key in ("event_count", "node_occurrences", "edge_count", "matched_endpoint_join_count"):
            totals[key] += int(case[key])
    totals["snapshot_count"] = global_snapshot
    audit["summary"] = {
        "scenario_count": 10, "host_stream_count": 16, **totals,
    }
    audit_path = output_dir / "atlas_v1_source_link_audit.json"
    audit_path.write_bytes(_canonical(audit))

    evidence = {row["scenario"]: row for row in audit["attack_evidence"]["scenarios"]}
    records = []
    for case in audit["cases"]:
        for join in case["matched_endpoint_joins"]:
            record = {
                "schema": "athena.atlas_v1.malicious_endpoint_join.v1",
                "record_type": "malicious_endpoint_join", "dataset": "atlas",
                "scene": case["scenario"], "host": case["host"],
                "source_id": "atlas-v1:" + hashlib.sha256(_canonical({
                    "scenario": case["scenario"], "host": case["host"],
                    "anchor": join["anchor"],
                })).hexdigest(),
                "official_labels": case["labels"], "resolved_anchor": join["anchor"],
                "matched_labels": join["matched_labels"],
                "matched_event_occurrence_count": join["matched_event_occurrence_count"],
                "boundary": {key: join[key] for key in (
                    "snapshot", "snapshot_local", "window_start", "event_id", "event_order",
                    "event_timestamp", "action", "endpoint_role", "anchor", "anchor_type",
                    "event_sha256",
                )},
                "source": case["source"],
                "paper_attack_features": evidence[case["scenario"]]["paper_features"],
                "attack_mapping_candidates": evidence[case["scenario"]]["attack_mapping_candidates"],
                "mapping_status": "human_candidates_separate_from_official_endpoint_ground_truth",
            }
            record["derived_sha256"] = hashlib.sha256(_canonical(record)).hexdigest()
            records.append(record)
    jsonl_path = output_dir / "atlas_v1_source_link_endpoints.jsonl"
    jsonl_path.write_text("".join(
        _canonical(row).decode("utf-8") + "\n" for row in records
    ), encoding="utf-8")
    manifest = {
        "schema": "athena.atlas_v1.source_link_manifest.v1",
        "record_count": len(records), "scenarios": list(SCENARIOS),
        "hosts": sorted({f"{row['scene']}:{row['host']}" for row in records}),
        "source_repository": audit["source_repository"],
        "paper_source": audit["attack_evidence"]["paper_source"],
        "archives": archives,
        "files": {
            audit_path.name: {"sha256": _sha256(audit_path), "size_bytes": audit_path.stat().st_size},
            jsonl_path.name: {"sha256": _sha256(jsonl_path), "size_bytes": jsonl_path.stat().st_size},
        },
        "parser_sources": {
            rel: _sha256(REPO_ROOT / rel) for rel in (
                "src/snapshot_construction/atlas_v1.py",
                "src/snapshot_construction/atlas_parser.py",
                "src/utils/interval_replay.py",
            )
        },
        "generator": {"filename": Path(__file__).name, "sha256": _sha256(Path(__file__))},
        "records": [{"source_id": row["source_id"], "derived_sha256": row["derived_sha256"]}
                    for row in records],
    }
    manifest["aggregate_records_sha256"] = hashlib.sha256(b"".join(
        _canonical(row) for row in manifest["records"]
    )).hexdigest()
    manifest_path = output_dir / "atlas_v1_source_link_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return {"audit": audit_path, "records": jsonl_path, "manifest": manifest_path,
            "record_count": len(records)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True,
                        help="unpacked official paper_experiments root")
    parser.add_argument("--zip-dir", type=Path, required=True,
                        help="directory containing official S1.zip...M6.zip blobs")
    parser.add_argument("--paper-pdf", type=Path, required=True,
                        help="official USENIX ATLAS paper PDF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", default="e46096d1947e4f059e73a0ac2b9a9707812fd4bc")
    args = parser.parse_args(argv)
    result = build(args.input_root, args.zip_dir, args.paper_pdf, args.output_dir,
                   args.source_commit)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
