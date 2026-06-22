"""Paper §IV.A + §IV.D — Train the encoder + classifier on a chosen dataset.

Replaces the in-tree paper-workspace train/test entrypoints. Loads the dataset
handler, builds 1-min provenance snapshots, fits the ATHENA encoder
(3-layer typed GIN + GRU temporal + hard-weighted contrastive loss), then
trains the 2-layer MLP detector on top of the snapshot embeddings.

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
    return p.parse_args(argv)


@measure_func("prepare_data", realtime=True, interval=1.0)
def prepare_data(path_map, dataset, scene):
    handler = get_handler(dataset, True, path_map, scene_name=scene)
    handler.load()
    handler.build_graph(dataset)
    return handler


@measure_func("train_encoder", realtime=True, interval=1.0)
def train_encoder(handler):
    encoder = ATHENAEncoder(handler.snapshots)
    encoder.train()
    return encoder.get_snapshot_embeddings()


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    path_map = cfg.get("paths", {})

    handler = prepare_data(path_map, args.dataset, args.scene)
    snapshot_embeddings = train_encoder(handler)

    benign_embeddings = snapshot_embeddings[
        handler.benign_idx_start:handler.benign_idx_end + 1
    ]
    detector = ATHENADetector()
    detector.train(benign_embeddings)
    print(
        f"[detection] dataset={args.dataset} scene={args.scene} "
        f"snapshot_count={len(snapshot_embeddings)} device={torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )


if __name__ == "__main__":
    main()
