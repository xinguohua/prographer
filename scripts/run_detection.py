"""Train the ATHENA encoder + node-level detector on a chosen dataset.

Loads the dataset handler, builds 1-min provenance snapshots, fits the ATHENA
encoder (3-layer typed GIN + GRU temporal + hard-weighted contrastive loss),
generates a per-node embedding, then trains the 2-layer MLP detector on
``(node embedding, malicious-UUID label)`` pairs.

Usage:
    python scripts/run_detection.py --config configs/athena.yaml --dataset cadets
    python scripts/run_detection.py --dataset atlas --scene M1-CVE-2015-5122
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection import ATHENAEncoder, ATHENADetector
from src.detection.node_labels import flatten_node_embeddings, load_malicious_uuids
from src.snapshot_construction.graph_loader import get_handler
from src.utils.config import load_config
from src.utils.io import measure_func


SUPPORTED_DATASETS = (
    "cadets", "theia", "trace", "clearscope",
    "cadets5", "theia5", "trace5", "clearscope5",
    "atlas", "optcday1",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "athena.yaml"),
                   help="path to athena.yaml")
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS,
                   help="dataset key from configs/athena.yaml::paths")
    p.add_argument("--scene", default=None,
                   help="optional scene filter (e.g. cadets314)")
    p.add_argument("--epochs", type=int, default=None,
                   help="override detection.epochs from the config")
    p.add_argument("--max-snapshots", type=int, default=None,
                   help="cap on snapshots used (smoke testing)")
    p.add_argument("--use-temporal", action="store_true",
                   help="use GRU temporal state when generating node embeddings")
    p.add_argument("--output", default="outputs/detection_predictions.json",
                   help="JSON path for node-level detector predictions")
    p.add_argument("--augmented-dir", default="outputs/augmented_graphs",
                   help="directory produced by run_augmentation.py; admitted graphs are used as contrastive hard negatives when present")
    return p.parse_args(argv)


@measure_func("prepare_data", realtime=True, interval=1.0)
def prepare_data(path_map, dataset, scene):
    handler = get_handler(dataset, True, path_map, scene_name=scene)
    handler.load()
    handler.build_graph(dataset)
    return handler


def apply_snapshot_cap(handler, max_snapshots: Optional[int]):
    if max_snapshots is None:
        return handler
    max_snapshots = int(max_snapshots)
    if max_snapshots <= 0:
        raise ValueError("--max-snapshots must be positive")
    handler.snapshots = handler.snapshots[:max_snapshots]
    last = len(handler.snapshots) - 1
    if last < 0:
        handler.benign_idx_end = -1
        handler.malicious_idx_start = -1
        handler.malicious_idx_end = -1
        return handler
    handler.benign_idx_end = min(int(getattr(handler, "benign_idx_end", -1)), last)
    if int(getattr(handler, "malicious_idx_start", -1)) > last:
        handler.malicious_idx_start = -1
        handler.malicious_idx_end = -1
    else:
        handler.malicious_idx_end = min(int(getattr(handler, "malicious_idx_end", -1)), last)
    return handler


@measure_func("train_encoder", realtime=True, interval=1.0)
def train_encoder(
    handler,
    train_snapshot_ids: Iterable[int],
    cfg: dict,
    mutation_map: Optional[dict] = None,
    use_temporal: bool = False,
):
    gin_cfg = cfg.get("gin", {})
    contrastive_cfg = cfg.get("contrastive", {})
    encoder = ATHENAEncoder(
        handler.snapshots,
        train_indices=sorted(set(train_snapshot_ids)),
        use_temporal=use_temporal,
        prop_feat_dim=int(gin_cfg.get("hidden_dim", 64)),
        enc_hidden_dim=int(gin_cfg.get("hidden_dim", 64)),
        enc_out_dim=int(gin_cfg.get("out_dim", gin_cfg.get("hidden_dim", 64))),
        gin_layers=int(gin_cfg.get("num_layers", 3)),
        dropout=float(gin_cfg.get("dropout", 0.1)),
        r_hop=int(gin_cfg.get("r_hop", 4)),
        temperature=float(contrastive_cfg.get("temperature", 0.10)),
        anomaly_alpha=float(contrastive_cfg.get("hard_weight", 2.0)),
    )
    if mutation_map:
        encoder.mutation_map = mutation_map
    encoder.train()
    encoder.generate_node_embeddings(use_temporal=use_temporal)
    return encoder


def _binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def _snapshot_range(start, end):
    if start is None or end is None:
        return []
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        return []
    return list(range(start, end + 1))


def _chronological_split(snapshot_ids, train_ratio: float):
    snapshot_ids = list(snapshot_ids)
    if not snapshot_ids:
        return [], []
    if len(snapshot_ids) == 1:
        return snapshot_ids, []
    cut = int(round(len(snapshot_ids) * float(train_ratio)))
    cut = max(1, min(cut, len(snapshot_ids) - 1))
    return snapshot_ids[:cut], snapshot_ids[cut:]


def _snapshot_time_key(handler, sid: int) -> tuple:
    snapshots = getattr(handler, "snapshots", [])
    if sid < 0 or sid >= len(snapshots):
        return (1, sid)
    g = snapshots[sid]
    values = []
    try:
        if g is not None and g.vcount() > 0 and "timestamp" in g.vs.attributes():
            values.extend(float(v) for v in g.vs["timestamp"] if v is not None)
    except Exception:
        pass
    try:
        if g is not None and g.ecount() > 0 and "timestamp" in g.es.attributes():
            values.extend(float(v) for v in g.es["timestamp"] if v is not None)
    except Exception:
        pass
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return (1, sid)
    return (0, min(values), sid)


def build_split(handler, train_ratio: float):
    all_ids = list(range(len(getattr(handler, "snapshots", []))))
    ordered_ids = sorted(all_ids, key=lambda sid: _snapshot_time_key(handler, sid))
    train_ids, test_ids = _chronological_split(ordered_ids, train_ratio)
    if not test_ids:
        test_ids = train_ids[:]

    return train_ids, test_ids, {
        "mode": "chronological_by_snapshot_timestamp",
        "train_ratio": float(train_ratio),
        "train_snapshots": train_ids,
        "test_snapshots": test_ids,
        "ordered_snapshots": ordered_ids,
    }


def load_augmented_graphs(path: Union[str, Path], allowed_anchor_ids: Optional[set[int]] = None):
    """Load admitted augmented graphs produced by ``run_augmentation.py``.

    Returns ``(mutation_map, metadata)`` where ``mutation_map`` is keyed by the
    benign anchor snapshot index expected by the ATHENA encoder.
    """
    aug_dir = Path(path)
    if not aug_dir.is_absolute():
        aug_dir = REPO_ROOT / aug_dir
    manifest_path = aug_dir / "manifest.json"
    metadata = {
        "path": str(aug_dir),
        "manifest": str(manifest_path),
        "loaded_graphs": 0,
        "filtered_graphs": 0,
        "available": False,
    }
    if not manifest_path.exists():
        return {}, metadata
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    mutation_map: dict[int, list] = {}
    for item in manifest.get("admitted", []):
        graph_name = item.get("graph")
        if not graph_name:
            continue
        graph_path = aug_dir / str(graph_name)
        if not graph_path.exists():
            continue
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
        anchor_sid = int(item.get("benign_snapshot", -1))
        if anchor_sid >= 0:
            if allowed_anchor_ids is not None and anchor_sid not in allowed_anchor_ids:
                metadata["filtered_graphs"] += 1
                continue
            mutation_map.setdefault(anchor_sid, []).append(graph)
            metadata["loaded_graphs"] += 1
    metadata["available"] = True
    metadata["manifest_admitted_count"] = int(manifest.get("admitted_count", 0))
    metadata["manifest_rejected_count"] = int(manifest.get("rejected_count", 0))
    metadata["anchors"] = sorted(int(k) for k in mutation_map)
    return mutation_map, metadata


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    path_map = cfg.get("paths", {})
    det_cfg = cfg.get("detection", {})
    split_mode = str(det_cfg.get("split_mode", "chronological_by_snapshot_timestamp"))
    if split_mode != "chronological_by_snapshot_timestamp":
        raise ValueError(f"unsupported detection.split_mode: {split_mode}")

    handler = prepare_data(path_map, args.dataset, args.scene)
    handler = apply_snapshot_cap(handler, args.max_snapshots)
    train_ratio = float(det_cfg.get("train_ratio", 0.70))
    train_snapshot_ids, test_snapshot_ids, split_meta = build_split(handler, train_ratio)
    print(
        f"[node-detection] split={split_meta['mode']} train_snapshots={len(train_snapshot_ids)} "
        f"test_snapshots={len(test_snapshot_ids)} train_ratio={train_ratio:.2f}"
    )

    mutation_map, augmentation_meta = load_augmented_graphs(
        args.augmented_dir,
        allowed_anchor_ids=set(train_snapshot_ids),
    )
    if mutation_map:
        print(
            f"[node-detection] loaded_augmented_graphs={augmentation_meta['loaded_graphs']} "
            f"anchors={len(augmentation_meta.get('anchors', []))} from={augmentation_meta['manifest']}"
        )
    else:
        print(
            f"[node-detection] no augmented graphs loaded from {augmentation_meta['manifest']}; "
            "training uses original snapshots only"
        )

    use_temporal = bool(det_cfg.get("use_temporal", True)) or bool(args.use_temporal)

    encoder = train_encoder(
        handler,
        train_snapshot_ids,
        cfg,
        mutation_map=mutation_map,
        use_temporal=use_temporal,
    )
    encoder_config = {
        "use_temporal": bool(encoder.use_temporal),
        "gin_layers": int(encoder.gin_layers),
        "embedding_dim": int(encoder.enc_out_dim),
        "hidden_dim": int(encoder.enc_hidden_dim),
        "r_hop": int(encoder.r_hop),
        "temperature": float(encoder.temperature),
        "hard_weight": float(encoder.anomaly_alpha),
        "train_snapshots": list(encoder.train_snapshot_indices),
    }

    malicious_uuids = load_malicious_uuids(args.dataset, args.scene or "")
    if not malicious_uuids:
        raise RuntimeError(
            f"no released malicious-entity labels found for dataset={args.dataset} scene={args.scene}; "
            "add the corresponding label file under data/annotated_labels before computing supervised metrics"
        )
    X, y, uuids, sids = flatten_node_embeddings(
        encoder.snapshot_node_embeddings, malicious_uuids=malicious_uuids,
    )

    n_total = int(X.shape[0])
    n_mal = int((y == 1).sum())
    n_ben = n_total - n_mal
    print(
        f"[node-detection] dataset={args.dataset} scene={args.scene} "
        f"nodes={n_total} benign={n_ben} malicious={n_mal} "
        f"labels_loaded={len(malicious_uuids)}"
    )

    if n_total == 0:
        print("[node-detection] no node embeddings generated; aborting")
        return

    epochs = args.epochs or int(det_cfg.get("epochs", 50))
    mlp_hidden = int(det_cfg.get("mlp_hidden", 256))
    train_snapshot_set = set(train_snapshot_ids)
    test_snapshot_set = set(test_snapshot_ids)
    train_mask = np.asarray([int(sid) in train_snapshot_set for sid in sids], dtype=bool)
    test_mask = np.asarray([int(sid) in test_snapshot_set for sid in sids], dtype=bool)
    if not bool(train_mask.any()):
        raise RuntimeError("no training nodes after chronological split")
    if not bool(test_mask.any()):
        test_mask = train_mask.copy()
    benign_train_mask = train_mask & (y == 0)
    malicious_train_mask = train_mask & (y == 1)
    detector = ATHENADetector(hidden_dim=mlp_hidden, num_epochs=epochs)
    if int(malicious_train_mask.sum()) > 0:
        detector.train(
            benign_embeddings=X[benign_train_mask],
            malicious_embeddings=X[malicious_train_mask],
            malicious_labels=y[malicious_train_mask],
        )
    else:
        detector.train(benign_embeddings=X[benign_train_mask])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[node-detection] device={device} "
        f"encoder={encoder.__class__.__name__} detector={detector.__class__.__name__}"
    )

    pred_labels, details = detector.predict(X)
    metrics = _binary_metrics(y[test_mask], pred_labels[test_mask])
    train_metrics = _binary_metrics(y[train_mask], pred_labels[train_mask])
    print(
        "[node-detection] test_metrics "
        f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} fpr={metrics['fpr']:.4f}"
    )

    predictions = []
    for pos, (uuid, sid, true_label, pred_label) in enumerate(zip(uuids, sids, y, pred_labels)):
        row = {
            "position": int(pos),
            "snapshot": int(sid),
            "uuid": str(uuid),
            "true_label": int(true_label),
            "pred_label": int(pred_label),
            "split": "test" if int(sid) in test_snapshot_set else "train",
        }
        if pos in details:
            row["prob_malicious"] = float(details[pos].get("prob_malicious", 0.0))
        predictions.append(row)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "scene": args.scene,
        "use_temporal": bool(use_temporal),
        "encoder_config": encoder_config,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "split": split_meta,
        "augmentation": augmentation_meta,
        "counts": {
            "nodes": n_total,
            "benign": n_ben,
            "malicious": n_mal,
            "train_nodes": int(train_mask.sum()),
            "test_nodes": int(test_mask.sum()),
            "labels_loaded": len(malicious_uuids),
        },
        "predictions": predictions,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[node-detection] wrote {out_path}")


if __name__ == "__main__":
    main()
