"""Global attack interpretation: technique mapping + tactic-level alignment.

For each snapshot marked malicious in the loaded dataset:
1. Extract the key causal sub-path rooted at every malicious node.
2. Translate the sub-path into a natural-language query
   (:mod:`src.interpretation.semantic_matching`).
3. Map the query to the most similar parent-level ATT&CK technique via
   Sentence-BERT cosine similarity over the technique knowledge base.
4. Aggregate the per-snapshot top-1 techniques into a sequence, fold them
   to the corresponding ATT&CK tactic sequence, and LCS-align against the
   curated attack-sequence library.

Usage:
    python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets --scene cadets314
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.detection.node_labels import load_malicious_uuids
from src.interpretation.attack_subgraph import extract_attack_subgraph, extract_key_path
from src.interpretation.global_alignment import lcs_min_ratio
from src.interpretation.semantic_matching import TechniqueSemanticMapper, snapshot_to_query
from src.interpretation.tactic_alignment import (
    best_tactic_match,
    load_tactic_sequence_library,
    load_tech_to_tactic,
    techniques_to_tactics,
)
from src.snapshot_construction.graph_loader import get_handler
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
                   help="override interpretation.topk_candidates from the config")
    p.add_argument("--max-malicious", type=int, default=None,
                   help="cap the number of malicious snapshots interpreted (smoke testing)")
    p.add_argument("--output", default=None,
                   help="optional JSON file to dump per-snapshot interpretation + alignment")
    p.add_argument("--detections", default=None,
                   help="JSON produced by scripts/run_detection.py; used as the interpretation input")
    p.add_argument("--use-ground-truth", action="store_true",
                   help="debug/evaluation mode: interpret released malicious UUIDs instead of detector output")
    return p.parse_args(argv)


def _malicious_nodes(snap, malicious_uuids: set) -> list:
    """Return igraph vertex indices that are malicious.

    A node is malicious if either its ``label`` attribute equals 1 (set by
    the augmented training pipeline on ego subgraphs) or its ``name`` (CDM
    UUID) is in the released malicious-entity set for the scene.
    """
    mal = []
    for v in range(snap.vcount()):
        attrs = snap.vs[v].attributes()
        try:
            if int(attrs.get("label", 0)) == 1:
                mal.append(v)
                continue
        except (TypeError, ValueError):
            pass
        name = str(attrs.get("name", ""))
        if name and name in malicious_uuids:
            mal.append(v)
    return mal


def _mark_malicious(snap, malicious_uuids: set) -> int:
    """Set ``label=1`` on every snapshot vertex whose ``name`` UUID is in the
    released malicious-entity set so downstream helpers that key off
    ``label`` (e.g. :func:`snapshot_to_query`) see them. Returns the count
    of vertices newly labelled."""
    if not malicious_uuids:
        return 0
    count = 0
    for v in range(snap.vcount()):
        attrs = snap.vs[v].attributes()
        name = str(attrs.get("name", ""))
        if name in malicious_uuids:
            try:
                if int(attrs.get("label", 0)) != 1:
                    snap.vs[v]["label"] = 1
                    count += 1
            except (TypeError, ValueError):
                snap.vs[v]["label"] = 1
                count += 1
    return count


def _load_detected_nodes(path: str) -> dict:
    """Load detector positives from ``scripts/run_detection.py`` output.

    Returns ``{snapshot_id: set(uuid)}``. Ground-truth labels in the detection
    file are ignored here; they are only for metric reporting.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for row in payload.get("predictions", []):
        if int(row.get("pred_label", 0)) != 1:
            continue
        sid = int(row["snapshot"])
        out.setdefault(sid, set()).add(str(row["uuid"]))
    return out


def _mark_detected(snap, detected_uuids: set) -> int:
    count = 0
    for v in range(snap.vcount()):
        attrs = snap.vs[v].attributes()
        name = str(attrs.get("name", ""))
        is_detected = name in detected_uuids
        snap.vs[v]["label"] = 1 if is_detected else int(attrs.get("label", 0) or 0)
        if is_detected:
            count += 1
    return count


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    interp_cfg = cfg.get("interpretation", {})
    top_k = args.top_k or int(interp_cfg.get("topk_candidates", 10))
    gamma = float(interp_cfg.get("gamma", 0.50))
    min_ratio = float(interp_cfg.get("lcs_min_ratio", 0.60))
    retention_days = int(interp_cfg.get("tactic_queue_retention_days", 7))

    if not args.use_ground_truth and not args.detections:
        raise SystemExit(
            "run_interpretation.py now requires --detections <run_detection.json>. "
            "Use --use-ground-truth only for annotation/debug evaluation."
        )

    handler = get_handler(args.dataset, True, cfg.get("paths", {}), scene_name=args.scene)
    handler.load()
    handler.build_graph(args.dataset)

    malicious_uuids = set()
    detected_by_snapshot = {}
    if args.use_ground_truth:
        mal_start = int(getattr(handler, "malicious_idx_start", -1) or -1)
        mal_end = int(getattr(handler, "malicious_idx_end", -1) or -1)
        if mal_start < 0 or mal_end < mal_start:
            print(f"[interpretation] no malicious snapshots in {args.dataset}/{args.scene}")
            return
        mal_indices = list(range(mal_start, mal_end + 1))
        malicious_uuids = load_malicious_uuids(args.dataset, args.scene or "")
    else:
        detected_by_snapshot = _load_detected_nodes(args.detections)
        mal_indices = sorted(detected_by_snapshot)
        if not mal_indices:
            print(f"[interpretation] detector produced no malicious nodes in {args.detections}")
            return

    if args.max_malicious is not None:
        mal_indices = mal_indices[: args.max_malicious]

    marked_total = 0
    for sidx in mal_indices:
        if args.use_ground_truth:
            marked_total += _mark_malicious(handler.snapshots[sidx], malicious_uuids)
        else:
            marked_total += _mark_detected(handler.snapshots[sidx], detected_by_snapshot.get(sidx, set()))

    triples_path = str(REPO_ROOT / "data" / "attack_knowledge"
                       / "mitre_attack" / "technique_triples_transformed.json")
    mapper = TechniqueSemanticMapper(
        triples_path=triples_path,
        encoder_name=str(interp_cfg.get("sentence_bert_model", "all-MiniLM-L12-v2")),
        gamma=gamma,
        top_k=top_k,
    )

    tactic_lib = load_tactic_sequence_library()
    tech_to_tactic = load_tech_to_tactic()

    print(
        f"[interpretation] dataset={args.dataset} scene={args.scene} "
        f"malicious_snapshots={len(mal_indices)} mapper_top_k={top_k} "
        f"gamma={gamma} tactic_queue_retention_days={retention_days} "
        f"lcs_min_ratio={min_ratio} tactic_lib_size={len(tactic_lib)} "
        f"input={'ground_truth' if args.use_ground_truth else args.detections} "
        f"labels_loaded={len(malicious_uuids)} nodes_marked={marked_total}"
    )

    per_snapshot = []
    technique_seq: list = []
    for sidx in mal_indices:
        snap = handler.snapshots[sidx]
        mal_nodes = _malicious_nodes(snap, malicious_uuids if args.use_ground_truth else set())
        if not mal_nodes:
            continue

        attack_sub = extract_attack_subgraph(snap, mal_nodes)
        key_path = extract_key_path(snap, mal_nodes)

        query = snapshot_to_query(attack_sub, node_scope="malicious")
        if not query:
            query = snapshot_to_query(snap, node_scope="malicious")
        if not query:
            continue

        ranked = mapper.predict_top_k_detail(query)
        if not ranked:
            continue
        top1 = ranked[0]
        tech_id = top1.get("tech_id", "")
        score = float(top1.get("score", 0.0))
        if not tech_id or score < gamma:
            continue

        technique_seq.append(tech_id)
        per_snapshot.append({
            "snapshot": sidx,
            "malicious_nodes": len(mal_nodes),
            "key_path_edges": len(key_path),
            "technique": tech_id,
            "tactic": tech_to_tactic.get(tech_id, "Unmapped"),
            "score": round(score, 4),
            "query_preview": query[:160],
        })

    tactic_seq = techniques_to_tactics(technique_seq, mapping=tech_to_tactic)
    best_lib_seq, best_score = best_tactic_match(
        tactic_seq, tactic_library=tactic_lib, min_ratio=min_ratio,
    )

    print(f"\n[interpretation] technique sequence (len={len(technique_seq)}): {technique_seq}")
    print(f"[interpretation] tactic    sequence (len={len(tactic_seq)}): {tactic_seq}")
    print(f"[interpretation] best library match: {best_lib_seq} (LCS/min={best_score:.2f}"
          f", min_ratio={min_ratio:.2f}, {'HIT' if best_lib_seq else 'NO HIT'})")
    if tactic_lib and tactic_seq:
        all_scores = [(ref, lcs_min_ratio(tactic_seq, ref)) for ref in tactic_lib if ref]
        all_scores.sort(key=lambda x: -x[1])
        print("[interpretation] LCS/min scores vs every library sequence:")
        for ref, r in all_scores:
            print(f"    ratio={r:.2f}  ref={ref}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": args.dataset,
            "scene": args.scene,
            "technique_sequence": technique_seq,
            "tactic_sequence": tactic_seq,
            "best_library_match": best_lib_seq,
            "best_lcs_min_ratio": best_score,
            "lcs_min_ratio": min_ratio,
            "tactic_queue_retention_days": retention_days,
            "input_mode": "ground_truth" if args.use_ground_truth else "detector_output",
            "detections": args.detections,
            "per_snapshot": per_snapshot,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[interpretation] wrote {out_path}")


if __name__ == "__main__":
    main()
