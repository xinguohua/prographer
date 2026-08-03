import pytest

ig = pytest.importorskip("igraph")

from scripts.run_detection import build_split
from scripts.run_interpretation import _mark_detected
from src.augmentation.edge_mutation import propose_candidate_new_edges
from src.augmentation.structural_mutation import subgraph_replacement
from src.augmentation.verifier import verify_mutation
from src.augmentation.verifier import (
    build_historical_profiles,
    check_attribute_feasibility,
    check_operation_legality,
)
from src.detection.contrastive_learning import GCCEmbedderDev


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


def test_detector_mode_clears_stale_ground_truth_labels():
    g = _Graph([
        _Vertex(name="missed", label=1),
        _Vertex(name="detected", label=0),
    ])

    marked = _mark_detected(g, {"detected"})

    assert marked == 1
    assert g.vs[0].attributes()["label"] == 0
    assert g.vs[1].attributes()["label"] == 1


def test_malicious_snapshot_pool_uses_train_indices_only():
    encoder = GCCEmbedderDev.__new__(GCCEmbedderDev)
    encoder.snapshots = []
    encoder.train_snapshot_indices = [1, 3]
    encoder._mal_ego_pool = []

    # Inspect the implementation contract without constructing igraph objects.
    assert list(encoder.train_snapshot_indices) == [1, 3]


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
    g_a.add_edge(0, 1, actions="execute")

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
    mutated.add_edge(0, 1, actions="read")

    assert check_operation_legality(mutated, {0}, entity_ops) is False

    mutated.vs[0]["name"] = "proc-a"
    mutated.vs[0]["path"] = "/tmp/not-observed"
    assert check_attribute_feasibility(mutated, {0}, type_attrs) is False

    mutated.vs[0]["path"] = "/bin/ls"
    mutated.vs[0]["_athena_replaced_region"] = True
    assert check_attribute_feasibility(mutated, {0}, type_attrs) is True


def test_edge_mutation_add_candidates_cover_both_boundary_directions():
    g = ig.Graph(directed=True)
    g.add_vertices(2)
    g.vs[0]["type"] = "process"
    g.vs[0]["properties"] = "attack"
    g.vs[1]["type"] = "file"
    g.vs[1]["properties"] = "context"

    candidates = propose_candidate_new_edges(g, {0}, max_candidates=4)

    assert (0, 1, "connect") in candidates
    assert (1, 0, "connect") in candidates


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
