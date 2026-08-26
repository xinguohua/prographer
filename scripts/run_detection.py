"""Train the ATHENA encoder + node-level detector on a chosen dataset.

Loads the dataset handler, builds 1-min provenance snapshots, fits the ATHENA
encoder (3-layer GIN + final-layer GRU + hard-weighted contrastive loss),
generates a per-node embedding, then trains the 2-layer MLP detector on
``(node embedding, malicious-UUID label)`` pairs.

Usage:
    python scripts/run_detection.py --config configs/athena.yaml --dataset cadets
"""
from __future__ import annotations

import argparse
import hashlib
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
from src.utils.split import SPLIT_MODE, build_split


SUPPORTED_DATASETS = (
    "cadets", "theia", "trace", "clearscope",
    "cadets5", "theia5", "trace5", "clearscope5",
    "optcday1",
    "atlas",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "athena.yaml"),
                   help="path to athena.yaml")
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS,
                   help="dataset key from configs/athena.yaml::paths")
    p.add_argument("--scene", default=None,
                   help="optional scene filter; for ATLAS, the held-out original fold S1-S4 or M1-M6")
    p.add_argument("--epochs", type=int, default=None,
                   help="override detection.epochs from the config")
    p.add_argument("--max-snapshots", type=int, default=None,
                   help="cap on snapshots used (smoke testing)")
    p.add_argument("--use-temporal", action="store_true",
                   help="use GRU temporal state when generating node embeddings")
    p.add_argument("--output", default="outputs/detection_predictions.json",
                   help="JSON path for node-level detector predictions")
    p.add_argument("--augmented-dir", default="outputs/augmented_graphs",
                   help="directory produced by run_augmentation.py")
    p.add_argument("--benign-injection-manifest", type=Path,
                   help="source-linked E3 24/48/72h replay manifest")
    p.add_argument(
        "--mode", choices=("complete", "ablation-no-augmentation"), default="complete",
        help="complete ATHENA requires split-bound admitted variants; the explicit ablation omits them",
    )
    p.add_argument("--execution", choices=("train-save", "eval-only"), default="train-save")
    p.add_argument("--checkpoint", type=Path, help="E3 train-save checkpoint for eval-only")
    p.add_argument("--checkpoint-out", type=Path, help="write frozen encoder/GRU/Word2Vec/MLP")
    return p.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_hash_contract(snapshots, split: dict) -> dict:
    from src.utils.interval_replay import graph_sha256
    ids = sorted({
        int(value) for key in ("train_snapshots", "test_snapshots")
        for value in split.get(key, [])
    })
    rows = {str(index): graph_sha256(snapshots[index]) for index in ids}
    aggregate = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"snapshots": rows, "aggregate_sha256": aggregate}


def _initial_split(handler, dataset: str, execution: str, train_ratio: float):
    """Return the target split without fitting or partitioning an E5 transfer stream."""
    if execution == "eval-only" and dataset in {"cadets5", "theia5", "trace5", "clearscope5"}:
        test_ids = list(range(len(handler.snapshots)))
        if not test_ids:
            raise RuntimeError("E5 transfer target stream contains no snapshots")
        return [], test_ids, {
            "mode": "e3-checkpoint-transfer-eval", "train_snapshots": [],
            "test_snapshots": test_ids, "target_stream": "all-e5-snapshots",
        }
    return build_split(handler, train_ratio)


def _save_checkpoint(
    path: Path, encoder, detector, dataset: str, scene, split: dict,
    source_run_mode: str, augmentation_contract: dict,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_contract = _snapshot_hash_contract(encoder.snapshots, split)
    torch.save({
        "source_dataset": dataset, "source_scene": scene, "source_split": split,
        "source_run_mode": source_run_mode, "source_variant": "full-athena",
        "seed": int(detector.cfg.seed),
        "source_augmentation": augmentation_contract,
        "source_snapshot_contract": snapshot_contract,
        "encoder_params": {
            "use_temporal": encoder.use_temporal, "prop_feat_dim": encoder.prop_feat_dim,
            "enc_hidden_dim": encoder.enc_hidden_dim, "enc_out_dim": encoder.enc_out_dim,
            "gin_layers": encoder.gin_layers, "dropout": encoder.dropout,
            "temperature": encoder.temperature, "r_hop": encoder.r_hop,
            "anomaly_alpha": encoder.anomaly_alpha,
            "seed": encoder.seed,
        },
        "encoder": encoder.encoder.state_dict(), "temporal": encoder.temporal.state_dict(),
        "w2v_model": encoder._w2v_model,
        "detector_config": detector.cfg.__dict__, "detector_input_dim": detector.input_dim,
        "detector": detector.model.state_dict(),
    }, path)
    return {"path": str(path.resolve()), "sha256": _sha256(path),
            "source_dataset": dataset, "source_scene": scene, "source_split": split,
            "source_snapshot_contract": snapshot_contract,
            "seed": int(detector.cfg.seed),
            "source_run_mode": source_run_mode, "source_variant": "full-athena",
            "source_augmentation": augmentation_contract}


def _load_checkpoint(path: Path, snapshots):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if len(snapshots) < 2:
        raise RuntimeError("eval-only requires at least two snapshots")
    params = dict(state["encoder_params"])
    encoder = ATHENAEncoder(
        snapshots, train_indices=[0], test_indices=list(range(1, len(snapshots))), **params,
    )
    encoder.encoder.load_state_dict(state["encoder"])
    encoder.temporal.load_state_dict(state["temporal"])
    encoder._w2v_model = state.get("w2v_model")
    encoder.generate_external_embeddings()
    detector = ATHENADetector(**{
        key: value for key, value in state.get("detector_config", {}).items()
        if key in {"hidden_dim", "dropout", "lr", "num_epochs", "batch_size", "seed"}
    })
    detector.input_dim = int(state["detector_input_dim"])
    detector.model = detector._build_model()
    detector.model.load_state_dict(state["detector"])
    detector.model.eval()
    provenance = {
        "path": str(path.resolve()), "sha256": _sha256(path),
        "source_dataset": state.get("source_dataset"),
        "source_scene": state.get("source_scene"),
        "source_split": state.get("source_split"),
        "source_snapshot_contract": state.get("source_snapshot_contract"),
        "source_run_mode": state.get("source_run_mode"),
        "source_variant": state.get("source_variant"),
        "source_augmentation": state.get("source_augmentation"),
        "seed": state.get("seed"),
    }
    return encoder, detector, provenance


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
    test_snapshot_ids: Iterable[int],
    cfg: dict,
    mutation_map: Optional[dict] = None,
    use_temporal: bool = False,
):
    gin_cfg = cfg.get("gin", {})
    contrastive_cfg = cfg.get("contrastive", {})
    encoder = ATHENAEncoder(
        handler.snapshots,
        train_indices=list(dict.fromkeys(int(value) for value in train_snapshot_ids)),
        test_indices=list(dict.fromkeys(int(value) for value in test_snapshot_ids)),
        use_temporal=use_temporal,
        prop_feat_dim=int(gin_cfg.get("hidden_dim", 64)),
        enc_hidden_dim=int(gin_cfg.get("hidden_dim", 64)),
        enc_out_dim=int(gin_cfg.get("out_dim", gin_cfg.get("hidden_dim", 64))),
        gin_layers=int(gin_cfg.get("num_layers", 3)),
        dropout=float(gin_cfg.get("dropout", 0.1)),
        r_hop=int(gin_cfg.get("r_hop", 4)),
        temperature=float(contrastive_cfg.get("temperature", 0.10)),
        anomaly_alpha=float(contrastive_cfg.get("beta", 2.0)),
        seed=int(cfg.get("detection", {}).get("seed", 42)),
    )
    if mutation_map:
        encoder.mutation_map = mutation_map
    encoder.train()
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


def load_augmented_graphs(
    path: Union[str, Path],
    allowed_anchor_ids: Optional[set[int]] = None,
    allowed_attack_ids: Optional[set[int]] = None,
    expected_split: Optional[dict] = None,
    expected_dataset: Optional[str] = None,
    expected_scene: Optional[str] = None,
):
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
        "filtered_anchor_graphs": 0,
        "filtered_attack_graphs": 0,
        "available": False,
    }
    if not manifest_path.exists():
        return {}, metadata
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if expected_dataset is not None and manifest.get("dataset") != expected_dataset:
        raise RuntimeError(
            f"augmentation dataset {manifest.get('dataset')!r} does not match {expected_dataset!r}"
        )
    if manifest.get("scene") != expected_scene:
        raise RuntimeError(
            f"augmentation scene {manifest.get('scene')!r} does not match {expected_scene!r}"
        )

    if expected_split is not None:
        contract = manifest.get("split_contract")
        if not isinstance(contract, dict):
            raise RuntimeError(
                f"augmentation manifest {manifest_path} predates the train-only donor contract; "
                "delete the directory and rerun scripts/run_augmentation.py"
            )
        if contract.get("donor_policy") != "train_only":
            raise RuntimeError(
                f"augmentation manifest {manifest_path} does not declare donor_policy=train_only"
            )
        expected_mode = str(expected_split.get("mode", ""))
        if str(contract.get("mode", "")) != expected_mode:
            raise RuntimeError(
                f"augmentation split mode {contract.get('mode')!r} does not match "
                f"detection split mode {expected_mode!r}"
            )
        for key in ("fold", "family", "train_scenarios", "test_scenarios"):
            expected_value = expected_split.get(key)
            if expected_value is not None and contract.get(key) != expected_value:
                raise RuntimeError(
                    f"augmentation split field {key}={contract.get(key)!r} does not match "
                    f"detection split field {expected_value!r}"
                )
        manifest_train = {int(value) for value in contract.get("train_snapshots", [])}
        manifest_test = {int(value) for value in contract.get("test_snapshots", [])}
        expected_train = {int(value) for value in expected_split.get("train_snapshots", [])}
        expected_test = {int(value) for value in expected_split.get("test_snapshots", [])}
        if manifest_train != expected_train or manifest_test != expected_test:
            raise RuntimeError(
                f"augmentation manifest {manifest_path} was generated with a different snapshot "
                "split; regenerate augmentations with the current dataset, scene, and config"
            )

    mutation_map: dict[int, list] = {}
    for item in manifest.get("admitted", []):
        graph_name = item.get("graph")
        if not graph_name:
            continue
        anchor_sid = int(item.get("benign_snapshot", -1))
        attack_sid = int(item.get("attack_snapshot", -1))
        if anchor_sid < 0 or attack_sid < 0:
            metadata["filtered_graphs"] += 1
            continue
        if allowed_anchor_ids is not None and anchor_sid not in allowed_anchor_ids:
            metadata["filtered_graphs"] += 1
            metadata["filtered_anchor_graphs"] += 1
            continue
        if allowed_attack_ids is not None and attack_sid not in allowed_attack_ids:
            metadata["filtered_graphs"] += 1
            metadata["filtered_attack_graphs"] += 1
            continue
        graph_path = aug_dir / str(graph_name)
        if not graph_path.exists():
            continue
        with graph_path.open("rb") as f:
            graph = pickle.load(f)
        mutation_map.setdefault(anchor_sid, []).append(graph)
        metadata["loaded_graphs"] += 1
    metadata["available"] = True
    metadata["manifest_admitted_count"] = int(manifest.get("admitted_count", 0))
    metadata["manifest_rejected_count"] = int(manifest.get("rejected_count", 0))
    metadata["manifest_sha256"] = _sha256(manifest_path)
    metadata["split_contract"] = manifest.get("split_contract")
    metadata["anchors"] = sorted(int(k) for k in mutation_map)
    return mutation_map, metadata


def validate_augmentation_mode(mode: str, mutation_map: dict, metadata: dict) -> None:
    """Prevent an incomplete run from being reported as the full ATHENA system."""
    if mode == "ablation-no-augmentation":
        if mutation_map:
            raise RuntimeError("no-augmentation ablation must not load augmented graphs")
        return
    if mode != "complete":
        raise ValueError(f"unknown detection mode: {mode}")
    if not metadata.get("available"):
        raise RuntimeError(
            "complete ATHENA requires an augmentation manifest generated for this split"
        )
    if int(metadata.get("manifest_admitted_count", 0)) <= 0:
        raise RuntimeError("complete ATHENA requires at least one admitted augmented variant")
    if int(metadata.get("loaded_graphs", 0)) <= 0 or not mutation_map:
        raise RuntimeError(
            "complete ATHENA loaded no split-valid augmented variants; rerun augmentation"
        )


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    path_map = cfg.get("paths", {})
    det_cfg = cfg.get("detection", {})
    seed = int(det_cfg.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    split_mode = str(det_cfg.get("split_mode", SPLIT_MODE))
    if args.dataset != "atlas" and split_mode != SPLIT_MODE:
        raise ValueError(f"unsupported detection.split_mode: {split_mode}")

    handler = prepare_data(path_map, args.dataset, args.scene)
    handler = apply_snapshot_cap(handler, args.max_snapshots)
    train_ratio = float(det_cfg.get("train_ratio", 0.70))
    e5_transfer = args.execution == "eval-only" and args.dataset in {"cadets5", "theia5", "trace5", "clearscope5"}
    if e5_transfer:
        train_snapshot_ids, test_snapshot_ids, split_meta = _initial_split(
            handler, args.dataset, args.execution, train_ratio,
        )
        base_split_meta = {}
        base_snapshot_contract = {}
    else:
        train_snapshot_ids, test_snapshot_ids, split_meta = _initial_split(
            handler, args.dataset, args.execution, train_ratio,
        )
        base_split_meta = json.loads(json.dumps(split_meta))
        base_snapshot_contract = _snapshot_hash_contract(handler.snapshots, base_split_meta)
        if not train_snapshot_ids:
            raise RuntimeError("date-partition split produced no training snapshots")
        if not test_snapshot_ids:
            raise RuntimeError("date-partition split produced no held-out attack snapshots")
    split_detail = (
        f"fold={split_meta['fold']} train_scenarios={split_meta['train_scenarios']}"
        if split_meta.get("fold")
        else f"train_ratio={train_ratio:.2f}"
    )
    print(
        f"[node-detection] split={split_meta['mode']} train_snapshots={len(train_snapshot_ids)} "
        f"test_snapshots={len(test_snapshot_ids)} {split_detail}"
    )

    if args.execution == "eval-only":
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise ValueError("eval-only requires --checkpoint from an E3 train-save run")
        mutation_map = {}
        augmentation_meta = {"available": False, "eval_only_frozen_training": True}
    elif args.mode == "complete":
        mutation_map, augmentation_meta = load_augmented_graphs(
            args.augmented_dir,
            allowed_anchor_ids=set(train_snapshot_ids),
            allowed_attack_ids=set(train_snapshot_ids),
            expected_split=split_meta,
            expected_dataset=args.dataset,
            expected_scene=args.scene,
        )
    else:
        mutation_map = {}
        aug_dir = Path(args.augmented_dir)
        if not aug_dir.is_absolute():
            aug_dir = REPO_ROOT / aug_dir
        augmentation_meta = {
            "path": str(aug_dir), "manifest": str(aug_dir / "manifest.json"),
            "available": False, "loaded_graphs": 0, "ablation_omitted": True,
        }
    if args.execution == "train-save":
        validate_augmentation_mode(args.mode, mutation_map, augmentation_meta)
    interval_replay_meta = None
    if args.benign_injection_manifest:
        if args.dataset not in {"cadets", "theia", "trace", "clearscope"}:
            raise ValueError("benign interval replay is defined only for DARPA E3")
        from src.utils.interval_replay import apply_interval_replay
        inserted_ids, interval_replay_meta = apply_interval_replay(
            handler, args.benign_injection_manifest, args.dataset, args.scene,
            expected_split=base_split_meta,
        )
        test_snapshot_ids = list(test_snapshot_ids) + inserted_ids
        split_meta = {**split_meta, "test_snapshots": list(test_snapshot_ids),
                      "interval_replay": interval_replay_meta}
    if mutation_map:
        print(
            f"[node-detection] loaded_augmented_graphs={augmentation_meta['loaded_graphs']} "
            f"anchors={len(augmentation_meta.get('anchors', []))} from={augmentation_meta['manifest']}"
        )
    elif args.mode == "ablation-no-augmentation":
        print(
            "[node-detection] explicit ablation=no-augmentation; training uses original snapshots only"
        )

    use_temporal = bool(det_cfg.get("use_temporal", True)) or bool(args.use_temporal)

    checkpoint_meta = None
    frozen_detector = None
    if args.execution == "eval-only":
        encoder, frozen_detector, checkpoint_meta = _load_checkpoint(
            args.checkpoint, handler.snapshots,
        )
        if checkpoint_meta.get("source_dataset") not in {"cadets", "theia", "trace", "clearscope"}:
            raise RuntimeError("eval-only checkpoint must originate from DARPA E3 training")
        if (
            checkpoint_meta.get("source_run_mode") != "complete"
            or checkpoint_meta.get("source_variant") != "full-athena"
            or not (checkpoint_meta.get("source_augmentation") or {}).get("manifest_sha256")
        ):
            raise RuntimeError("eval-only requires a complete ATHENA E3 checkpoint")
        if args.dataset in {"cadets", "theia", "trace", "clearscope"} and (
            checkpoint_meta.get("source_dataset") != args.dataset
            or checkpoint_meta.get("source_scene") != args.scene
            or (checkpoint_meta.get("source_split") or {}).get("train_snapshots")
            != base_split_meta.get("train_snapshots")
            or (checkpoint_meta.get("source_split") or {}).get("test_snapshots")
            != base_split_meta.get("test_snapshots")
            or checkpoint_meta.get("source_snapshot_contract") != base_snapshot_contract
        ):
            raise RuntimeError("E3 interval replay must reuse its own Basic dataset/scene checkpoint")
        e5_source = {
            "cadets5": "cadets", "theia5": "theia",
            "trace5": "trace", "clearscope5": "clearscope",
        }.get(args.dataset)
        if e5_source and checkpoint_meta.get("source_dataset") != e5_source:
            raise RuntimeError(f"{args.dataset} transfer requires a {e5_source} E3 checkpoint")
        target_test_ids = (
            list(test_snapshot_ids)
            if args.dataset in {"cadets", "theia", "trace", "clearscope"}
            else list(range(len(handler.snapshots)))
        )
        train_snapshot_ids = []
        test_snapshot_ids = target_test_ids
        split_meta = {
            "mode": "e3-checkpoint-transfer-eval",
            "train_snapshots": [], "test_snapshots": test_snapshot_ids,
            "source_training": checkpoint_meta,
            "interval_replay": interval_replay_meta,
        }
    else:
        encoder = train_encoder(
            handler,
            train_snapshot_ids,
            test_snapshot_ids,
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
        "beta": float(encoder.anomaly_alpha),
        "train_snapshots": list(train_snapshot_ids),
        "test_snapshots": list(test_snapshot_ids),
    }

    malicious_uuids = load_malicious_uuids(args.dataset, args.scene or "")
    malicious_uuids.update(str(value) for value in getattr(handler, "all_labels", []))
    snapshot_malicious_uuids = []
    for snapshot_id, graph in enumerate(getattr(handler, "snapshots", [])):
        if graph is None or "label" not in graph.vs.attributes():
            raise RuntimeError(
                f"snapshot {snapshot_id} lacks parser-scoped node labels; "
                "paper-profile detection does not fall back to a cross-host label union"
            )
        graph_malicious = {
            str(graph.vs[index]["name"])
            for index, label in enumerate(graph.vs["label"])
            if int(label or 0) == 1
        }
        snapshot_malicious_uuids.append(graph_malicious)
        malicious_uuids.update(graph_malicious)
    if not malicious_uuids:
        raise RuntimeError(
            f"no released malicious-entity labels found for dataset={args.dataset} scene={args.scene}; "
            "add the corresponding label file under data/annotated_labels before computing supervised metrics"
        )
    X, y, uuids, sids = flatten_node_embeddings(
        encoder.snapshot_node_embeddings, malicious_uuids=malicious_uuids,
        snapshot_malicious_uuids=snapshot_malicious_uuids,
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

    epochs = args.epochs or int(det_cfg.get("epochs", 3))
    mlp_hidden = int(det_cfg.get("mlp_hidden", 256))
    train_snapshot_set = set(train_snapshot_ids)
    test_snapshot_set = set(test_snapshot_ids)
    train_mask = np.asarray([int(sid) in train_snapshot_set for sid in sids], dtype=bool)
    test_mask = np.asarray([int(sid) in test_snapshot_set for sid in sids], dtype=bool)
    if args.execution == "train-save" and not bool(train_mask.any()):
        raise RuntimeError("no training nodes after date-partition split")
    if not bool(test_mask.any()):
        raise RuntimeError("no held-out test nodes after date-partition split")
    if args.execution == "train-save":
        benign_train_mask = train_mask & (y == 0)
        malicious_train_mask = train_mask & (y == 1)
        if int(benign_train_mask.sum()) == 0 or int(malicious_train_mask.sum()) == 0:
            raise RuntimeError(
                "detector training requires both benign and malicious nodes in the training split"
            )
        detector = ATHENADetector(hidden_dim=mlp_hidden, num_epochs=epochs, seed=seed)
        detector.train(
            benign_embeddings=X[benign_train_mask],
            malicious_embeddings=X[malicious_train_mask],
            malicious_labels=y[malicious_train_mask],
        )
        if args.checkpoint_out:
            checkpoint_meta = _save_checkpoint(
                args.checkpoint_out, encoder, detector, args.dataset, args.scene, split_meta,
                args.mode, {
                    "manifest_sha256": augmentation_meta.get("manifest_sha256"),
                    "split_contract": augmentation_meta.get("split_contract"),
                    "admitted_count": augmentation_meta.get("manifest_admitted_count"),
                },
            )
    else:
        detector = frozen_detector

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[node-detection] device={device} "
        f"encoder={encoder.__class__.__name__} detector={detector.__class__.__name__}"
    )

    pred_labels, details = detector.predict(X)
    metrics = _binary_metrics(y[test_mask], pred_labels[test_mask])
    train_metrics = (
        _binary_metrics(y[train_mask], pred_labels[train_mask])
        if bool(train_mask.any()) else None
    )
    print(
        "[node-detection] test_metrics "
        f"acc={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} fpr={metrics['fpr']:.4f}"
    )

    predictions = []
    for pos, (uuid, sid, true_label, pred_label) in enumerate(zip(uuids, sids, y, pred_labels)):
        if args.execution == "eval-only" and int(sid) not in test_snapshot_set:
            continue
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
        "variant": "full-athena" if args.mode == "complete" else "ablation-no-augmentation",
        "run_mode": args.mode,
        "execution": args.execution,
        "seed": int(checkpoint_meta.get("seed", seed) if checkpoint_meta else seed),
        "checkpoint": checkpoint_meta,
        "use_temporal": bool(encoder.use_temporal),
        "encoder_config": encoder_config,
        "metrics": metrics,
        "train_metrics": train_metrics,
        "split": split_meta,
        "augmentation": augmentation_meta,
        "interval_replay": interval_replay_meta,
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
