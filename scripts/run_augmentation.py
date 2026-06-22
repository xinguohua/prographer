"""Paper §IV.B + §IV.C — LLM-guided graph augmentation.

For each benign anchor in the requested dataset:
1. Retrieve Top-K similar attack snapshots via the WL subtree kernel
   (:func:`src.augmentation.subgraph_retrieval.top_k_similar_attacks`).
2. Search for an aligned region S' (Algorithm 1) and replace it into the
   anchor (:mod:`src.augmentation.structural_mutation`).
3. Apply LLM-guided boundary edge mutation
   (:mod:`src.augmentation.edge_mutation`, supp B).
4. Apply LLM-guided semantic mutation (replacement / rewriting / extension
   in :mod:`src.augmentation.semantic_mutation`, supp C.1–C.3).
5. Submit the mutated graph through the four unified-verification checks
   (:mod:`src.augmentation.verifier`); admit only mutations that pass the
   two hard checks (imperceptibility, hardness).

Usage:
    python scripts/run_augmentation.py --config configs/athena.yaml --dataset cadets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.snapshot_construction.graph_loader import get_handler
from src.augmentation.subgraph_retrieval import top_k_similar_attacks
from src.augmentation.structural_mutation import aligned_region_search, subgraph_replacement
from src.augmentation.edge_mutation import apply_edge_mutation_llm
from src.augmentation.semantic_mutation import (
    apply_semantic_mutation_llm,
    generate_strategy_variants,
)
from src.augmentation.verifier import verify_mutation, build_historical_profiles
from src.utils.config import load_config


SUPPORTED_DATASETS = (
    "cadets", "theia", "trace", "clearscope",
    "cadets5", "theia5",
    "atlas", "optcday1",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "athena.yaml"))
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p.add_argument("--scene", default=None)
    p.add_argument("--top-k", type=int, default=None,
                   help="override augmentation.top_k from the config")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    aug_cfg = cfg.get("augmentation", {})
    top_k = args.top_k or int(aug_cfg.get("top_k", 5))
    top_m = int(aug_cfg.get("top_m", 3))
    delta_h = float(aug_cfg.get("delta_h_lower", 0.30))
    delta_h_upper = float(aug_cfg.get("delta_h_upper", 0.95))

    handler = get_handler(args.dataset, True, cfg.get("paths", {}), scene_name=args.scene)
    handler.load()
    handler.build_graph(args.dataset)

    benign_graphs = [(handler.snapshots[i], i)
                     for i in range(handler.benign_idx_start, handler.benign_idx_end + 1)]
    attack_graphs = [(handler.snapshots[i], i)
                     for i in range(handler.attack_idx_start, handler.attack_idx_end + 1)]

    entity_ops, type_attrs = build_historical_profiles(benign_graphs)

    admitted = 0
    rejected = 0
    for g_anchor, b_idx in benign_graphs:
        candidates = top_k_similar_attacks(g_anchor, attack_graphs, k=top_k)
        for g_attack, _ in candidates:
            regions = aligned_region_search(g_anchor, g_attack)
            if not regions:
                continue
            for benign_region, attack_region, _mapping, _score in regions[:top_m]:
                g_mut, replaced = subgraph_replacement(
                    g_anchor, g_attack, benign_region, attack_region
                )
                g_mut, _edge_actions = apply_edge_mutation_llm(g_mut, replaced)
                g_mut = apply_semantic_mutation_llm(g_mut, replaced)
                passed, _failed = verify_mutation(
                    g_mut, g_anchor, replaced, entity_ops, type_attrs,
                    delta_h=delta_h, delta_h_upper=delta_h_upper,
                )
                if passed:
                    admitted += 1
                else:
                    rejected += 1

    print(f"[augmentation] dataset={args.dataset} admitted={admitted} rejected={rejected}")


if __name__ == "__main__":
    main()
