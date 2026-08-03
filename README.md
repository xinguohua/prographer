# ATHENA

ATHENA is a provenance-based intrusion detection system for stealthy Advanced Persistent Threats (APTs). It builds time-windowed provenance snapshots from system audit logs, learns discriminative node representations through LLM-guided adaptive contrastive learning, performs **per-node binary benign/malicious detection**, and reconstructs interpretable multi-stage attack chains by mapping each malicious node to an ATT&CK technique and **aligning the resulting tactic sequence** against multi-stage attack patterns.

The online supplementary material lives at <https://xinguohua.github.io/athena-supp/>.

## Method Overview

ATHENA has four stages:

1. **Snapshot construction.** Audit events are partitioned into 1-minute non-overlapping windows; each window is materialized as a typed provenance graph and decomposed into node-centred *r*-hop subgraphs. → `src/snapshot_construction/`.
2. **LLM-guided graph augmentation.** For each benign anchor, structurally similar attack subgraphs are retrieved via the Weisfeiler–Leman subtree kernel, an LLM-guided edge mutation decides which boundary edges between the substituted attack region and the surrounding context to ADD / REMOVE / KEEP, and three semantic-mutation strategies rewrite the attack process node's command name + arguments to blend into the benign context. A unified verification step filters mutations that violate operation legality, attribute feasibility, imperceptibility, or hardness. → `src/augmentation/`.
3. **Adaptive contrastive learning and node-level detection.** A 3-layer typed GIN with per-layer GRU temporal state is trained with a hard-sample-weighted supervised contrastive loss. A 2-layer MLP head consumes the per-node embedding produced by the encoder and emits a benign/malicious label for each node. → `src/detection/`.
4. **Technique mapping and tactic-level alignment.** Each detector-flagged node is mapped to a parent-level MITRE ATT&CK technique via Sentence-BERT similarity over the technique knowledge base; the techniques inside the same snapshot are aggregated into a persistent tactic queue and aligned against the multi-stage tactic-sequence library using the paper's LCS/min criterion. → `src/interpretation/`.

## Repository Layout

```
.
├── configs/athena.yaml         # paths + hyperparameters
├── prompts/                    # LLM prompt templates for augmentation
├── data/
│   ├── annotated_labels/       # released malicious-node + ATT&CK labels (E3, E5, OpTC, ATLAS)
│   └── attack_knowledge/       # ATT&CK technique KB + tactic-sequence library
├── src/
│   ├── snapshot_construction/  # 1-min provenance snapshots + r-hop ego subgraphs
│   ├── augmentation/           # WL retrieval + structural / semantic / edge mutation + verifier
│   ├── detection/              # typed GIN + GRU encoder, contrastive loss, node MLP head
│   ├── interpretation/         # node → technique mapping → tactic sequence → LCS/min alignment
│   └── utils/                  # config loader, timing helpers, LLM client
└── scripts/
    ├── run_augmentation.py     # writes admitted augmented graphs + manifest
    ├── run_detection.py        # writes held-out per-node predictions + metrics
    └── run_interpretation.py   # consumes detector output for tactic-sequence alignment
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.9 on Ubuntu, CUDA-capable GPU.

## Data Preparation

Download the upstream datasets and unpack them locally:

| Dataset | Source |
|---|---|
| DARPA E3 (Trace, Theia, Cadets, ClearScope) | <https://github.com/darpa-i2o/Transparent-Computing> |
| DARPA E5 (Trace, Theia, Cadets, ClearScope) | <https://github.com/darpa-i2o/Transparent-Computing-5D> |
| DARPA OpTC | <https://github.com/FiveDirections/OpTC-data> |
| ATLAS | <https://github.com/purseclab/ATLAS> |

Edit `configs/athena.yaml::paths` so each `<DATASET_ROOT>` placeholder points at the unpacked dataset directory.

Raw audit logs themselves are not redistributed here; consult each dataset's license.

## Released Labels (supp G.1)

`data/annotated_labels/` contains manually verified malicious-entity labels and ATT&CK technique labels for DARPA E3, DARPA E5, OpTC, and ATLAS, annotated by three doctoral researchers in system security based on the official attack reports. The labels are used for supervised training and metric computation; interpretation consumes detector outputs by default.

- `<dataset>/malicious_entities/` — one CDM-record UUID per line per scene, consumed by `collect_label_paths` in `src/snapshot_construction/_common.py`.
- `<dataset>/attack_techniques/` — per-scene UUID → parent-level MITRE ATT&CK technique + tactic mapping.

## ATT&CK Knowledge Base (supp G.2 v)

- `data/attack_knowledge/mitre_attack/technique_triples_{raw,transformed}.json` — operation-level action triples for each ATT&CK technique, used by `src/interpretation/semantic_matching.py` as the retrieval corpus.
- `data/attack_knowledge/attackseqbench/technique_sequences.txt` — released multi-stage attack-sequence sample library, used by `src/interpretation/global_alignment.py` for the paper's LCS/min alignment. Replace or extend this file with a larger sequence library when running broader sequence-retrieval studies.

## Prompt Registry (supp G.2 ii)

The four prompt templates in `prompts/` are loaded by the augmentation pipeline:

| Template | Used by | Supp section |
|---|---|---|
| `edge_mutation.txt`  | `src/augmentation/edge_mutation.py`     | B   |
| `replacement.txt`    | `src/augmentation/semantic_mutation.py` | C.1 |
| `rewriting.txt`      | `src/augmentation/semantic_mutation.py` | C.2 |
| `extension.txt`      | `src/augmentation/semantic_mutation.py` | C.3 |

## Running the Pipeline

```bash
# Augmentation (paper §IV.B + §IV.C): writes outputs/augmented_graphs/
python scripts/run_augmentation.py   --config configs/athena.yaml --dataset cadets

# Detection (paper §IV.A + §IV.D): writes held-out predictions and metrics
python scripts/run_detection.py      --config configs/athena.yaml --dataset cadets \
  --output outputs/detection_predictions.json

# Interpretation (paper §IV.E): consumes detector positives, not ground truth
python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets \
  --detections outputs/detection_predictions.json
```

Supported `--dataset` values: `cadets, theia, trace, clearscope` (DARPA E3); `cadets5, theia5` (DARPA E5); `optcday1` (OpTC day 1); `atlas` (ATLAS).

Optional flags: `--scene <name>` to filter a specific scene (e.g. `cadets314`); `--epochs N` and `--max-snapshots N` to constrain runs. The detection script follows the paper's chronological protocol by default: snapshots are kept in construction order, each benign/attack block is split by `detection.train_ratio`, the encoder and MLP are trained only on training snapshots, and `metrics` in the output JSON are computed on held-out test snapshots. `train_metrics` and the exact snapshot split are included for auditability.

## Configuration

Copy the local-secrets template and fill in your LLM API key:

```bash
cp local_settings.example.py local_settings.py
# Edit local_settings.py: CHATANYWHERE_API_KEY, CHATANYWHERE_ENDPOINT
```

`local_settings.py` is gitignored.

## Hyperparameters (supp D)

Defaults in `configs/athena.yaml` match the artifact configuration used by the released scripts:

| Symbol | Meaning | Default | Section |
|---|---|---|---|
| T | snapshot window | 1 min | snapshot |
| r | r-hop neighbourhood radius | 4 | gin |
| L | GIN layers | 3 | gin |
| D | embedding dimension | 64 | gin |
| top_k | WL retrieval candidates | 5 | augmentation |
| top_m | accepted mutations per anchor | 3 | augmentation |
| δ_h | WL similarity range (hardness check) | [0.30, 0.95] | augmentation |
| γ | mapping confidence cutoff | 0.50 | interpretation |
| train ratio | chronological detector training split | 0.70 | detection |
| T_max | tactic queue retention window | 7 days | interpretation |
| LCS/min | sequence-alignment cutoff | 0.60 | interpretation |

## Reproducibility Notes

- Seeds are set in `src/utils/seed.py` (call before training).
- LLM stochasticity is bounded by `temperature` in `configs/athena.yaml::llm`; runs against hosted LLMs are not bit-exact reproducible.
- `scripts/run_interpretation.py --use-ground-truth` is provided only for annotation/debug checks. The default paper pipeline requires `--detections` from `scripts/run_detection.py`.

## Citation

```bibtex
@inproceedings{athena2026,
  title  = {ATHENA: Interpretable Stealthy APT Detection via LLM-Augmented Graph Contrastive Learning and Semantic Provenance Abstraction},
  author = {<author list>},
  year   = {2026},
  note   = {Artifact: https://github.com/xinguohua/athena}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Released annotation labels follow the redistribution terms of the upstream datasets they annotate.
