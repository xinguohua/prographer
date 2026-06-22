"""LLM-guided graph augmentation pipeline.

For each benign anchor in the requested dataset:
1. WL subtree-kernel Top-K retrieval of structurally similar attack snapshots.
2. BFS-based aligned-region search + subgraph replacement.
3. LLM-guided boundary edge mutation (ADD / REMOVE / KEEP).
4. LLM-guided semantic mutation (replacement / rewriting / extension).
5. Unified verification (operation legality / attribute feasibility /
   imperceptibility / hardness).

Usage:
    python scripts/run_augmentation.py --config configs/athena.yaml --dataset cadets --scene cadets314
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
    _collect_benign_corpus,
)
from src.augmentation.verifier import verify_mutation, build_historical_profiles
from src.utils.config import load_config
from src.utils.llm import chatanywhere_summarize


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
    p.add_argument("--max-anchors", type=int, default=None,
                   help="cap on benign anchors processed (smoke testing)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip LLM-guided edge/semantic mutation (structural only)")
    p.add_argument("--mock-llm", action="store_true",
                   help="use a fake LLM that returns an empty JSON list (exercises code path without API calls)")
    return p.parse_args(argv)


def _mock_llm_fn(_prompt: str) -> str:
    return "[]"


def _build_llm_fn(model_name: str):
    """Return a ``llm_fn(prompt) -> str`` using ``local_settings`` credentials."""
    try:
        import local_settings  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing local_settings.py — copy local_settings.example.py and fill in your API key."
        ) from exc

    api_key = getattr(local_settings, "CHATANYWHERE_API_KEY", None)
    endpoint = getattr(local_settings, "CHATANYWHERE_ENDPOINT",
                       "https://api.chatanywhere.tech/v1/chat/completions")
    if not api_key or api_key.startswith("PASTE"):
        raise RuntimeError("CHATANYWHERE_API_KEY not set in local_settings.py")

    def llm_fn(prompt: str) -> str:
        return chatanywhere_summarize(
            text=prompt,
            api_key=api_key,
            endpoint=endpoint,
            model=model_name,
            timeout=60,
        ) or ""
    return llm_fn


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    aug_cfg = cfg.get("augmentation", {})
    llm_cfg = cfg.get("llm", {})
    top_k = args.top_k or int(aug_cfg.get("top_k", 5))
    top_m = int(aug_cfg.get("top_m", 3))
    delta_h_lower = float(aug_cfg.get("delta_h_lower", 0.30))
    delta_h_upper = float(aug_cfg.get("delta_h_upper", 0.95))
    model_name = llm_cfg.get("default", "gpt-4o")

    handler = get_handler(args.dataset, True, cfg.get("paths", {}), scene_name=args.scene)
    handler.load()
    handler.build_graph(args.dataset)

    benign_graphs = [(handler.snapshots[i], i)
                     for i in range(handler.benign_idx_start, handler.benign_idx_end + 1)]
    attack_graphs = [(handler.snapshots[i], i)
                     for i in range(handler.malicious_idx_start, handler.malicious_idx_end + 1)]

    if args.max_anchors:
        benign_graphs = benign_graphs[:args.max_anchors]

    entity_ops, type_attrs = build_historical_profiles(benign_graphs)
    benign_commands, benign_args_set, _benign_files = _collect_benign_corpus(benign_graphs)
    if args.no_llm:
        llm_fn = None
        llm_tag = "off"
    elif args.mock_llm:
        llm_fn = _mock_llm_fn
        llm_tag = "mock"
    else:
        llm_fn = _build_llm_fn(model_name)
        llm_tag = model_name

    print(
        f"[augmentation] dataset={args.dataset} scene={args.scene} "
        f"benign={len(benign_graphs)} attack={len(attack_graphs)} "
        f"llm={llm_tag} top_k={top_k} top_m={top_m}"
    )

    admitted = 0
    rejected = 0
    for g_anchor, b_idx in benign_graphs:
        candidates = top_k_similar_attacks(g_anchor, attack_graphs, k=top_k)
        for g_attack, _attack_idx, _sim in candidates:
            regions = aligned_region_search(g_anchor, g_attack)
            if not regions:
                continue
            for benign_region, attack_region, pi_map, _score in regions[:top_m]:
                g_mut = subgraph_replacement(
                    g_anchor, g_attack, benign_region, attack_region, pi_map,
                )
                if g_mut is None:
                    continue
                replaced = set(benign_region)
                g_mut, _edge_actions = apply_edge_mutation_llm(
                    g_mut, replaced, llm_fn=llm_fn,
                )
                g_mut = apply_semantic_mutation_llm(
                    g_mut,
                    attack_node_indices=list(replaced),
                    benign_commands=benign_commands,
                    benign_args=benign_args_set,
                    llm_fn=llm_fn,
                    model_name=model_name,
                )
                passed, _failed = verify_mutation(
                    g_mut, g_anchor, replaced, entity_ops, type_attrs,
                    delta_h=delta_h_lower, delta_h_upper=delta_h_upper,
                )
                if passed:
                    admitted += 1
                else:
                    rejected += 1

    print(f"[augmentation] dataset={args.dataset} admitted={admitted} rejected={rejected}")


if __name__ == "__main__":
    main()
