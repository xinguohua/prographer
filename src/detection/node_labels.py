"""Load released malicious-entity UUIDs for node-level detection.

DARPA E3 labels are selected by scene, DARPA E5 labels by platform attack
sources, and OpTC labels by paper-profile host. These entity labels are
distinct from the exact event/anchor boundaries used by interpretation.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np

from src.snapshot_construction._common import load_released_malicious_uuids

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASET_LABEL_DIR = {
    "cadets":     "darpa_e3",
    "theia":      "darpa_e3",
    "trace":      "darpa_e3",
    "clearscope": "darpa_e3",
    "cadets5":    "darpa_e5",
    "theia5":     "darpa_e5",
    "trace5":     "darpa_e5",
    "clearscope5": "darpa_e5",
    "optcday1":   "optc",
}

DATASET_SCENE_PREFIX = {
    "cadets": ("cadets",),
    "theia": ("theia",),
    "trace": ("trace",),
    "clearscope": ("clearscope",),
    "cadets5": ("cadets",),
    "theia5": ("theia",),
    "trace5": ("trace",),
    "clearscope5": ("clearscope",),
    "optcday1": ("host_",),
}


def _scene_to_basename(dataset: str, scene: str) -> str:
    if dataset == "optcday1":
        return f"host_{scene}"
    return scene


def load_malicious_uuids(dataset: str, scene: str) -> Set[str]:
    """Return the malicious-entity UUID set admitted for this dataset scope.

    E3 accepts a scene-specific file. E5 uses the explicit PIDSMaker attack
    sources registered for the selected platform. OpTC preserves host-local
    label sets inside the snapshot handler.
    """
    if dataset in {"cadets5", "theia5", "trace5", "clearscope5", "optcday1"}:
        return load_released_malicious_uuids(dataset, scene or None)

    label_dir = DATASET_LABEL_DIR.get(dataset)
    if label_dir is None:
        return set()
    root = REPO_ROOT / "data" / "annotated_labels" / label_dir / "malicious_entities"
    paths = []
    if dataset == "optcday1":
        # The manuscript profile evaluates only H051/H201/H501. Keep metric
        # labels on the same host scope enforced by the OpTC parser.
        paths = [
            root / "host_0051.txt",
            root / "host_0201.txt",
            root / "host_0501.txt",
        ]
    elif scene:
        base = _scene_to_basename(dataset, scene)
        paths = [root / f"{base}.txt", root / f"{base}.csv"]
    else:
        prefixes = DATASET_SCENE_PREFIX.get(dataset, ("",))
        paths = sorted(
            p for p in list(root.glob("*.txt")) + list(root.glob("*.csv"))
            if any(p.stem.startswith(prefix) for prefix in prefixes)
        )
    uuids = set()
    for p in paths:
        if not p.exists():
            continue
        if p.suffix.lower() == ".csv":
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in ("actorID", "objectID", "uuid", "name"):
                        value = (row.get(key) or "").strip()
                        if value:
                            uuids.add(value)
        else:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        uuids.add(t)
    return uuids


def flatten_node_embeddings(
    snapshot_node_embeddings: Iterable[dict],
    malicious_uuids: Optional[Set[str]] = None,
    snapshot_malicious_uuids: Optional[Iterable[Optional[Set[str]]]] = None,
):
    """Flatten ``List[Dict[uuid, embedding]]`` into ``(X, y, uuids, snap_ids)``.

    Args:
        snapshot_node_embeddings: output of
            :meth:`ATHENAEncoder.generate_node_embeddings`.
        malicious_uuids: set of UUIDs flagged as malicious; if ``None``, all
            labels are 0.

    Returns:
        X: ``(N, D)`` ``float32`` embedding matrix.
        y: ``(N,)`` ``int64`` binary label vector (1 = malicious).
        uuids: ``(N,)`` list of node UUIDs in row order.
        snap_ids: ``(N,)`` list of snapshot indices, parallel to ``X``.
    """
    mal = malicious_uuids or set()
    local_labels = list(snapshot_malicious_uuids) if snapshot_malicious_uuids is not None else None
    Xs, ys, us, sids = [], [], [], []
    for sidx, emb_dict in enumerate(snapshot_node_embeddings):
        if not emb_dict:
            continue
        snapshot_mal = (
            local_labels[sidx]
            if local_labels is not None and sidx < len(local_labels) and local_labels[sidx] is not None
            else mal
        )
        for nid, emb in emb_dict.items():
            Xs.append(emb)
            ys.append(1 if nid in snapshot_mal else 0)
            us.append(nid)
            sids.append(sidx)
    if not Xs:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [], [],
        )
    return (
        np.stack(Xs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        us,
        sids,
    )
