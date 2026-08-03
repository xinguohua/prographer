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
import sys
from pathlib import Path

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
    "cadets5", "theia5",
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
    return p.parse_args(argv)


@measure_func("prepare_data", realtime=True, interval=1.0)
def prepare_data(path_map, dataset, scene):
    handler = get_handler(dataset, True, path_map, scene_name=scene)
    handler.load()
    handler.build_graph(dataset)
    return handler


@measure_func("train_encoder", realtime=True, interval=1.0)
def train_encoder(handler, use_temporal: bool = False):
    encoder = ATHENAEncoder(handler.snapshots)
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


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    path_map = cfg.get("paths", {})

    handler = prepare_data(path_map, args.dataset, args.scene)
    encoder = train_encoder(handler, use_temporal=args.use_temporal)

    malicious_uuids = load_malicious_uuids(args.dataset, args.scene or "")
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

    epochs = args.epochs or int(cfg.get("detection", {}).get("epochs", 50))
    mlp_hidden = int(cfg.get("detection", {}).get("mlp_hidden", 256))
    benign_mask = y == 0
    detector = ATHENADetector(hidden_dim=mlp_hidden, num_epochs=epochs)
    if n_mal > 0:
        detector.train(
            benign_embeddings=X[benign_mask],
            malicious_embeddings=X[~benign_mask],
            malicious_labels=y[~benign_mask],
        )
    else:
        detector.train(benign_embeddings=X[benign_mask])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[node-detection] device={device} "
        f"encoder={encoder.__class__.__name__} detector={detector.__class__.__name__}"
    )

    pred_labels, details = detector.predict(X)
    metrics = _binary_metrics(y, pred_labels)
    print(
        "[node-detection] metrics "
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
        "use_temporal": bool(args.use_temporal),
        "metrics": metrics,
        "counts": {
            "nodes": n_total,
            "benign": n_ben,
            "malicious": n_mal,
            "labels_loaded": len(malicious_uuids),
        },
        "predictions": predictions,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[node-detection] wrote {out_path}")


if __name__ == "__main__":
    main()
