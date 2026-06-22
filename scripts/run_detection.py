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
import sys
from pathlib import Path

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


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    path_map = cfg.get("paths", {})

    handler = prepare_data(path_map, args.dataset, args.scene)
    encoder = train_encoder(handler, use_temporal=args.use_temporal)

    malicious_uuids = load_malicious_uuids(args.dataset, args.scene or "")
    X, y, _uuids, _sids = flatten_node_embeddings(
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

    benign_mask = y == 0
    detector = ATHENADetector()
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


if __name__ == "__main__":
    main()
