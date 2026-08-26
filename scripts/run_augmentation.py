"""LLM-guided graph augmentation pipeline.

For each benign anchor in the requested dataset:
1. WL subtree-kernel Top-K retrieval of structurally similar attack snapshots.
2. BFS-based aligned-region search + subgraph replacement.
3. LLM-guided boundary edge mutation (ADD / REMOVE / KEEP).
4. LLM-guided semantic mutation (replacement / rewriting / extension).
5. Unified verification (operation legality / attribute feasibility /
   imperceptibility / hardness).

Usage:
    python scripts/run_augmentation.py --config configs/athena.yaml --dataset cadets
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.snapshot_construction.graph_loader import get_handler
from src.augmentation.graph_units import build_graph_units
from src.augmentation.subgraph_retrieval import top_k_similar_attacks
from src.augmentation.subgraph_retrieval import wl_kernel
from src.augmentation.structural_mutation import aligned_region_search, subgraph_replacement
from src.augmentation.edge_mutation import apply_edge_mutation_llm
from src.augmentation.semantic_mutation import (
    apply_semantic_mutation_llm,
    _collect_benign_corpus,
)
from src.augmentation.verifier import (
    build_historical_profiles,
    check_attack_chain_fidelity,
    collect_audited_nodes,
    imperceptibility_coverage,
    verify_mutation,
)
from src.utils.config import load_config
from src.utils.llm import TrackedOpenAICompatibleLLM, summarize_llm_calls
from src.utils.split import SPLIT_MODE, build_split


SUPPORTED_DATASETS = (
    "cadets", "theia", "trace", "clearscope",
    "cadets5", "theia5", "trace5", "clearscope5",
    "optcday1",
    "atlas",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "athena.yaml"))
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    p.add_argument(
        "--scene", default=None,
        help="scene filter; for ATLAS, the held-out original fold S1-S4 or M1-M6",
    )
    p.add_argument("--top-k", type=int, default=None,
                   help="override augmentation.top_k from the config")
    p.add_argument("--model", default=None,
                   help="LLM model key from configs/athena.yaml::llm.models")
    p.add_argument("--max-anchors", type=int, default=None,
                   help="cap on benign anchors processed (smoke testing)")
    p.add_argument("--output-dir", default="outputs/augmented_graphs",
                   help="directory for admitted augmented graphs and manifest.json")
    return p.parse_args(argv)


def _build_llm_fn(model_name: str, model_cfg: dict):
    """Build the selected model's routed, telemetry-recording client."""
    provider = str(model_cfg.get("provider", "")).strip()
    base_url = str(model_cfg.get("base_url", "")).strip()
    base_url_env = str(model_cfg.get("base_url_env", "")).strip()
    if base_url_env:
        base_url = str(os.getenv(base_url_env, "")).strip()
    key_env = str(model_cfg.get("api_key_env", "")).strip()
    api_key = os.getenv(key_env, "") if key_env else ""
    if not api_key and bool(model_cfg.get("api_key_optional", False)):
        api_key = "EMPTY"
    if not provider or not base_url:
        raise RuntimeError(f"LLM model {model_name} requires provider and base_url")
    if not api_key:
        raise RuntimeError(f"LLM model {model_name} requires environment variable {key_env}")
    temperature = float(model_cfg.get("temperature", 0.2))
    top_p = float(model_cfg.get("top_p", 0.95))
    top_k = model_cfg.get("top_k")
    max_tokens = int(model_cfg.get("mutation_max_tokens", model_cfg.get("max_tokens", 1024)))
    stop = model_cfg.get("stop_tokens")
    served_model_env = str(model_cfg.get("served_model_env", "")).strip()
    served_model = (
        str(os.getenv(served_model_env, "")).strip()
        if served_model_env else str(model_cfg.get("served_model", model_cfg.get("version", model_name)))
    )
    if not served_model:
        raise RuntimeError(
            f"LLM model {model_name} requires environment variable {served_model_env}"
        )
    return TrackedOpenAICompatibleLLM(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=served_model,
        temperature=temperature,
        top_p=top_p,
        top_k=int(top_k) if top_k is not None else None,
        max_tokens=max_tokens,
        stop=stop,
        timeout=float(model_cfg.get("timeout_seconds", 60)),
        max_api_retries=int(model_cfg.get("max_api_retries", 0)),
        context_window=int(model_cfg.get("context_window", 4096)),
    )


def _replaced_region_nodes(g) -> set[int]:
    if g is None or g.vcount() == 0:
        return set()
    attrs = set(g.vs.attributes())
    if "_athena_replaced_region" not in attrs:
        return set()
    return {
        int(idx)
        for idx, flag in enumerate(g.vs["_athena_replaced_region"])
        if bool(flag)
    }


def _region_contains_attack_anchor(g, region) -> bool:
    """Require the selected S' to retain the attack graph's labelled centre."""
    if g is None or "_athena_anchor" not in g.vs.attributes():
        return False
    selected = {int(index) for index in region}
    return any(
        index in selected
        and bool(g.vs[index].attributes().get("_athena_anchor", False))
        and int(g.vs[index].attributes().get("label", 0) or 0) == 1
        for index in range(g.vcount())
    )


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    aug_cfg = cfg.get("augmentation", {})
    llm_cfg = cfg.get("llm", {})
    top_k = args.top_k or int(aug_cfg.get("top_k", 5))
    max_retries = int(aug_cfg.get("max_retries", 3))
    delta_h_lower = float(aug_cfg.get("delta_h_lower", 0.30))
    r_hop = int(cfg.get("gin", {}).get("r_hop", 4))
    model_name = args.model or llm_cfg.get("default", "gpt-4o")
    model_cfg = (llm_cfg.get("models", {}) or {}).get(model_name, {})
    if model_name not in (llm_cfg.get("models", {}) or {}):
        raise ValueError(f"unknown LLM model key: {model_name}")
    context_window = int(model_cfg.get("context_window", 4096))
    max_output_tokens = int(model_cfg.get("mutation_max_tokens", 1024))
    conservative_prompt_chars = max(512, (context_window - max_output_tokens - 64) * 3)
    edge_prompt_chars = min(
        int(aug_cfg.get("edge_prompt_char_budget", conservative_prompt_chars)),
        conservative_prompt_chars,
    )
    semantic_prompt_chars = min(
        int(aug_cfg.get("semantic_prompt_char_budget", conservative_prompt_chars)),
        conservative_prompt_chars,
    )

    handler = get_handler(args.dataset, True, cfg.get("paths", {}), scene_name=args.scene)
    handler.load()
    handler.build_graph(args.dataset)
    detection_cfg = cfg.get("detection", {})
    split_mode = str(detection_cfg.get("split_mode", SPLIT_MODE))
    if args.dataset != "atlas" and split_mode != SPLIT_MODE:
        raise ValueError(f"unsupported detection.split_mode: {split_mode}")
    train_ratio = float(detection_cfg.get("train_ratio", 0.70))
    train_snapshot_ids, test_snapshot_ids, split_meta = build_split(handler, train_ratio)
    if not test_snapshot_ids:
        raise RuntimeError("date-partition split produced no held-out attack-day snapshots")
    benign_graphs, attack_graphs = build_graph_units(
        handler, train_snapshot_ids, r_hop,
        max_cached_units=int(aug_cfg.get("graph_unit_cache_size", 32)),
    )
    if not attack_graphs:
        raise RuntimeError("no train-only attack graph units are available for augmentation")

    total_benign_anchors = len(benign_graphs)
    if args.max_anchors:
        benign_graphs = benign_graphs[:args.max_anchors]

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    training_graphs = [(handler.snapshots[sid], sid) for sid in train_snapshot_ids]
    entity_ops, type_attrs = build_historical_profiles(training_graphs)
    benign_commands, benign_args_set, _benign_files = _collect_benign_corpus(training_graphs)
    llm_fn = _build_llm_fn(model_name, model_cfg)
    llm_tag = model_name

    print(
        f"[augmentation] dataset={args.dataset} scene={args.scene} "
        f"benign={len(benign_graphs)} attack={len(attack_graphs)} "
        f"llm={llm_tag} top_k={top_k} r_hop={r_hop} "
        f"train_snapshots={len(train_snapshot_ids)}"
    )

    admitted = 0
    rejected = 0
    manifest = {
        "dataset": args.dataset,
        "scene": args.scene,
        "top_k": top_k,
        "max_retries": max_retries,
        "delta_h_lower": delta_h_lower,
        "r_hop": r_hop,
        "llm": llm_tag,
        "llm_config": {
            "provider": str(model_cfg.get("provider", "")),
            "base_url": str(model_cfg.get("base_url", "")),
            "base_url_env": str(model_cfg.get("base_url_env", "")),
            "api_key_env": str(model_cfg.get("api_key_env", "")),
            "served_model_env": str(model_cfg.get("served_model_env", "")),
            "resolved_model": str(llm_fn.model),
            "resolved_endpoint": str(llm_fn.endpoint),
            "precision": str(model_cfg.get("precision", "")),
            "temperature": float(model_cfg.get("temperature", 0.2)),
            "top_p": float(model_cfg.get("top_p", 0.95)),
            "top_k": int(model_cfg["top_k"]) if model_cfg.get("top_k") is not None else None,
            "max_tokens": int(model_cfg.get("mutation_max_tokens", 1024)),
            "stop_tokens": model_cfg.get("stop_tokens"),
            "pricing": model_cfg.get("pricing", {}) or {},
        },
        "split_contract": {
            "mode": split_meta["mode"],
            "train_ratio": split_meta["train_ratio"],
            "fold": split_meta.get("fold"),
            "family": split_meta.get("family"),
            "train_scenarios": split_meta.get("train_scenarios"),
            "test_scenarios": split_meta.get("test_scenarios"),
            "donor_policy": "train_only",
            "train_snapshots": split_meta["train_snapshots"],
            "test_snapshots": split_meta["test_snapshots"],
        },
        "admitted": [],
        "rejected": 0,
        "anchor_processing": {
            "total_available": total_benign_anchors,
            "scheduled": len(benign_graphs),
            "sampling": "deterministic_prefix" if args.max_anchors else "all",
            "processed": 0,
        },
    }
    attack_wl_cache = {}
    for g_anchor, benign_ref in benign_graphs:
        manifest["anchor_processing"]["processed"] += 1
        candidates = top_k_similar_attacks(
            g_anchor, attack_graphs, k=top_k, _attack_hist_cache=attack_wl_cache,
        )
        aligned_candidates = []
        for g_attack, attack_ref, retrieval_similarity in candidates:
            region = aligned_region_search(g_anchor, g_attack, r_hop=r_hop)
            attack_region = set(region[1]) if region is not None else set()
            contains_attack_anchor = _region_contains_attack_anchor(g_attack, attack_region)
            if region is not None and contains_attack_anchor:
                aligned_candidates.append((region[3], retrieval_similarity, g_attack, attack_ref, region))
        if not aligned_candidates:
            rejected += 1
            manifest["rejected"] += 1
            manifest.setdefault("rejection_details", []).append({
                "benign_snapshot": benign_ref.snapshot_id,
                "benign_anchor_node": benign_ref.anchor_node,
                "failed_checks": ["aligned_region_search_or_attack_fidelity"],
            })
            continue

        score, retrieval_similarity, g_attack, attack_ref, region = max(
            aligned_candidates, key=lambda item: item[0]
        )
        benign_region, attack_region, pi_map, _ = region
        accepted = False
        last_failed = []
        for attempt in range(1, max_retries + 1):
            g_mut = subgraph_replacement(
                g_anchor, g_attack, benign_region, attack_region, pi_map,
                r_hop=r_hop,
            )
            if g_mut is None:
                last_failed = ["structural_replacement"]
                break
            g_mut["dataset"] = args.dataset
            replaced = _replaced_region_nodes(g_mut)
            if not replaced:
                last_failed = ["structural_replacement_region"]
                break
            g_mut, _edge_actions = apply_edge_mutation_llm(
                g_mut,
                replaced,
                llm_fn=lambda prompt, current_attempt=attempt, **metadata: llm_fn(
                    prompt,
                    stage="edge_mutation",
                    attempt=current_attempt,
                    metadata=metadata,
                ),
                max_candidates_per_call=int(aug_cfg.get("edge_candidate_batch_size", 32)),
                max_prompt_chars=edge_prompt_chars,
            )
            if g_mut is None:
                last_failed = ["edge_mutation"]
                continue
            g_mut = apply_semantic_mutation_llm(
                g_mut,
                attack_node_indices=list(replaced),
                benign_commands=benign_commands,
                benign_args=benign_args_set,
                llm_fn=lambda prompt, current_attempt=attempt, **metadata: llm_fn(
                    prompt,
                    stage="semantic_mutation",
                    attempt=current_attempt,
                    metadata=metadata,
                ),
                model_name=model_name,
                r_hop=r_hop,
                max_prompt_chars=semantic_prompt_chars,
            )
            if g_mut is None:
                last_failed = ["semantic_mutation"]
                continue
            semantic_modified = {
                index for index in range(g_mut.vcount())
                if bool(g_mut.vs[index].attributes().get("_athena_semantic_modified", False))
            }
            verified_nodes = collect_audited_nodes(g_mut, set(replaced))
            fidelity_ok, fidelity_report = check_attack_chain_fidelity(g_mut)
            if not fidelity_ok:
                last_failed = ["attack_chain_fidelity"]
                continue
            passed, failed = verify_mutation(
                g_mut, g_anchor, verified_nodes, entity_ops, type_attrs,
                delta_h=delta_h_lower,
            )
            last_failed = failed
            if passed:
                imperceptibility_report = imperceptibility_coverage(g_mut)
                hardness_similarity = float(wl_kernel(g_mut, g_anchor, h=3))
                audited_edge_ids = [
                    int(edge.index) for edge in g_mut.es
                    if any(bool(edge.attributes().get(flag, False)) for flag in (
                        "_athena_boundary_edge", "_athena_edge_mutated", "_athena_introduced_edge",
                    ))
                ]
                admitted += 1
                graph_name = (
                    f"{args.dataset}_{args.scene or 'all'}_"
                    f"bs{benign_ref.snapshot_id}_bn{benign_ref.anchor_node}_"
                    f"as{attack_ref.snapshot_id}_an{attack_ref.anchor_node}.pkl"
                )
                graph_path = out_dir / graph_name
                with graph_path.open("wb") as f:
                    pickle.dump(g_mut, f)
                manifest["admitted"].append({
                    "graph": graph_name,
                    "benign_snapshot": benign_ref.snapshot_id,
                    "benign_anchor_node": benign_ref.anchor_node,
                    "benign_anchor_name": benign_ref.anchor_name,
                    "attack_snapshot": attack_ref.snapshot_id,
                    "attack_anchor_node": attack_ref.anchor_node,
                    "attack_anchor_name": attack_ref.anchor_name,
                    "retrieval_similarity": float(retrieval_similarity),
                    "mean_pair_similarity": float(score),
                    "replaced_nodes": sorted(int(x) for x in replaced),
                    "semantically_modified_nodes": sorted(int(x) for x in semantic_modified),
                    "verified_nodes": sorted(int(x) for x in verified_nodes),
                    "semantic_mutations": [
                        {
                            "node": int(index),
                            "strategy": str(g_mut.vs[index].attributes().get("_athena_semantic_strategy", "")),
                            "before": str(g_mut.vs[index].attributes().get("_athena_semantic_before", "")),
                            "after": str(g_mut.vs[index].attributes().get("_athena_semantic_after", "")),
                            "changed_fields": list(
                                g_mut.vs[index].attributes().get("_athena_semantic_changed_fields", [])
                            ),
                        }
                        for index in sorted(semantic_modified)
                        if g_mut.vs[index].attributes().get("_athena_semantic_strategy")
                    ],
                    "propagated_attribute_changes": [
                        {
                            "node": int(index),
                            "type": str(g_mut.vs[index].attributes().get("type", "")),
                            "before": str(g_mut.vs[index].attributes().get("_athena_semantic_before", "")),
                            "after": str(g_mut.vs[index].attributes().get("_athena_semantic_after", "")),
                            "changed_fields": list(
                                g_mut.vs[index].attributes().get("_athena_semantic_changed_fields", [])
                            ),
                        }
                        for index in sorted(semantic_modified)
                        if not g_mut.vs[index].attributes().get("_athena_semantic_strategy")
                    ],
                    "edge_actions": list(_edge_actions),
                    "attack_chain_fidelity": fidelity_report,
                    "imperceptibility_coverage": imperceptibility_report,
                    "verification_evidence": {
                        "operation_legality": {
                            "profile_keys": len(entity_ops),
                            "audited_edges": audited_edge_ids,
                        },
                        "attribute_feasibility": {
                            "profile_types": sorted(type_attrs),
                            "audited_nodes": sorted(int(value) for value in verified_nodes),
                        },
                        "imperceptibility": imperceptibility_report,
                        "hardness": {
                            "wl_similarity": hardness_similarity,
                            "minimum": delta_h_lower,
                        },
                    },
                    "attempt": int(attempt),
                })
                accepted = True
                break
        if not accepted:
            rejected += 1
            manifest["rejected"] += 1
            manifest.setdefault("rejection_details", []).append({
                "benign_snapshot": benign_ref.snapshot_id,
                "benign_anchor_node": benign_ref.anchor_node,
                "attack_snapshot": attack_ref.snapshot_id,
                "attack_anchor_node": attack_ref.anchor_node,
                "failed_checks": list(last_failed),
            })

    print(f"[augmentation] dataset={args.dataset} admitted={admitted} rejected={rejected}")
    manifest["admitted_count"] = admitted
    manifest["rejected_count"] = rejected
    manifest["llm_calls"] = list(llm_fn.records)
    manifest["retrieval_index"] = {
        "attack_units": len(attack_graphs),
        "cached_wl_histograms": len(attack_wl_cache),
        "cache_scope": "run",
    }
    pricing = model_cfg.get("pricing", {}) or {}
    pricing_by_model = {
        str(llm_fn.model): pricing
    }
    manifest["llm_usage_summary"] = summarize_llm_calls(
        manifest["llm_calls"], pricing_by_model,
    )
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[augmentation] wrote {manifest_path}")


if __name__ == "__main__":
    main()
