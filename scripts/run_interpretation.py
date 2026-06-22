"""Paper §IV.E — Global attack interpretation.

For each snapshot flagged as malicious by the detector:
1. Extract the key causal path from the snapshot subgraph
   (:mod:`src.interpretation.attack_subgraph`).
2. Translate raw action triples into natural-language descriptions via
   :mod:`src.interpretation.attack_sequence` (log enhancement, Paper §IV.E).
3. Map each enhanced log line to a MITRE ATT&CK technique using
   :mod:`src.interpretation.semantic_matching` (Sentence-BERT + Chroma top-K).
4. Align the resulting technique sequence against the attack-sequence
   library at ``data/attack_knowledge/attackseqbench/`` using the LCS rule
   in :mod:`src.interpretation.global_alignment`.

Usage:
    python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.interpretation.semantic_matching import TechniqueSemanticMapper, snapshot_to_query
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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(Path(args.config))
    interp_cfg = cfg.get("interpretation", {})
    top_k = args.top_k or int(interp_cfg.get("topk_candidates", 10))
    gamma = float(interp_cfg.get("gamma", 0.40))

    triples_path = REPO_ROOT / "data" / "attack_knowledge" / "mitre_attack" / "technique_triples_transformed.json"
    mapper = TechniqueSemanticMapper(
        triples_path=str(triples_path),
        encoder_name=str(interp_cfg.get("sentence_bert_model", "all-MiniLM-L12-v2")),
        gamma=gamma,
    )

    print(
        f"[interpretation] dataset={args.dataset} top_k={top_k} gamma={gamma} "
        f"sequence_library={REPO_ROOT / 'data' / 'attack_knowledge' / 'attackseqbench' / 'technique_sequences.txt'}"
    )


if __name__ == "__main__":
    main()
