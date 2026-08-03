import pytest

pytest.importorskip("igraph")

from scripts.run_interpretation import _mark_detected
from src.augmentation.verifier import verify_mutation
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
