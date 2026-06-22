# ATHENA

ATHENA is a provenance-based intrusion detection system for stealthy Advanced Persistent Threats (APTs). It builds time-windowed snapshots from system audit logs, learns discriminative graph representations through LLM-guided adaptive contrastive learning, and reconstructs interpretable multi-stage attack chains by aligning anomalous behaviours against MITRE ATT&CK techniques.

## Repository Scope

This repository contains the ATHENA implementation and released annotation resources. It does **not** include baseline implementations, paper plotting artifacts, full experiment logs, or raw audit logs restricted by dataset licenses.

The online supplementary material lives at <https://xinguohua.github.io/athena-supp/>.

## Method Overview

ATHENA has four stages (paper §IV.A–E):

1. **Snapshot construction (§IV.A).** Audit events are partitioned into 1-minute non-overlapping windows; each window is materialized as a typed provenance graph and decomposed into node-centred *r*-hop subgraphs. → `src/snapshot_construction/`.
2. **LLM-guided graph augmentation (§IV.B – §IV.C).** For each benign anchor, structurally similar attack subgraphs are retrieved via the Weisfeiler–Leman subtree kernel, an LLM-guided edge mutation decides which boundary edges between the substituted attack region and the surrounding context to ADD / REMOVE / KEEP, and three semantic-mutation strategies rewrite the attack process node's command name + arguments to blend into the benign context. A unified verification step filters mutations that violate operation legality, attribute feasibility, imperceptibility, or hardness. → `src/augmentation/`.
3. **Adaptive contrastive learning (§IV.D).** A 3-layer typed GIN with per-layer GRU temporal state is trained with a hard-sample-weighted supervised contrastive loss; a 2-layer MLP head produces per-snapshot binary anomaly labels. → `src/detection/`.
4. **Global attack interpretation (§IV.E).** Key causal paths are extracted from malicious snapshots, both sides are semantically enhanced, paths are mapped to ATT&CK techniques via Sentence-BERT similarity over the technique knowledge base, and the resulting technique sequence is aligned against multi-stage attack patterns using LCS. → `src/interpretation/`.

## Repository Layout

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── configs/
│   └── athena.yaml                       # paths + hyperparameters
├── prompts/                              # supp G.2 (ii) — prompt registry
│   ├── edge_mutation.txt                 # supp B
│   ├── replacement.txt                   # supp C.1
│   ├── rewriting.txt                     # supp C.2
│   └── extension.txt                     # supp C.3
├── data/
│   ├── annotated_labels/                 # supp G.1 — released labels
│   │   ├── darpa_e3/{malicious_entities,attack_techniques}/
│   │   ├── darpa_e5/{malicious_entities,attack_techniques}/
│   │   ├── optc/{malicious_entities,attack_techniques}/
│   │   └── atlas/
│   │       ├── malicious_entities/M1-CVE-2015-5122_windows_h{1,2}.csv
│   │       └── attack_techniques/groundtruth.txt
│   └── attack_knowledge/                 # supp G.2 (v) — ATT&CK KB
│       ├── mitre_attack/
│       │   ├── technique_triples_raw.json
│       │   └── technique_triples_transformed.json
│       └── attackseqbench/
│           └── technique_sequences.txt
├── src/                                  # supp G.2 — source code
│   ├── snapshot_construction/
│   │   ├── graph_loader.py               # dataset-key dispatch (handler_map + get_handler)
│   │   ├── snapshot_builder.py           # 1-min window partitioning + r-hop ego subgraphs
│   │   ├── darpa_e3_parser.py            # paper Table II: DARPA E3
│   │   ├── darpa_e5_parser.py            # paper Table II: DARPA E5
│   │   ├── optc_parser.py                # paper Table II: OpTC
│   │   └── atlas_parser.py               # paper Table II: ATLAS
│   ├── augmentation/
│   │   ├── subgraph_retrieval.py         # WL subtree kernel Top-K (supp A step 1)
│   │   ├── structural_mutation.py        # Algorithm 1 — aligned region + replacement
│   │   ├── edge_mutation.py              # supp B — LLM-guided ADD/REMOVE/KEEP
│   │   ├── semantic_mutation.py          # supp C.1–C.3 — three strategies
│   │   └── verifier.py                   # four unified-verification checks
│   ├── detection/
│   │   ├── gin_encoder.py                # 3-layer typed GIN
│   │   ├── temporal_encoder.py           # per-layer GRU
│   │   ├── contrastive_learning.py       # hard-weighted SupCon + train loop
│   │   └── classifier.py                 # 2-layer MLP head
│   ├── interpretation/
│   │   ├── attack_subgraph.py            # key causal path extraction
│   │   ├── semantic_matching.py          # Sentence-BERT + Chroma top-K
│   │   ├── attack_sequence.py            # log enhancement (action triples → NL)
│   │   └── global_alignment.py           # LCS alignment vs sequence library
│   └── utils/
│       ├── config.py                     # YAML loader for configs/athena.yaml
│       ├── io.py                         # timing/throughput helpers
│       └── llm.py                        # OpenAI-compatible LLM client
└── scripts/
    ├── run_augmentation.py
    ├── run_detection.py
    └── run_interpretation.py
```

## Paper-to-Code Mapping

| Paper section | Component | Path |
|---|---|---|
| §IV.A | 1-min snapshot partitioning, *r*-hop ego subgraphs | `src/snapshot_construction/snapshot_builder.py` |
| §IV.A | DARPA E3 / E5 / OpTC / ATLAS parsers | `src/snapshot_construction/{darpa_e3,darpa_e5,optc,atlas}_parser.py` |
| §IV.B | WL subtree-kernel Top-K retrieval | `src/augmentation/subgraph_retrieval.py` |
| §IV.B | BFS alignment + subgraph replacement (Algorithm 1) | `src/augmentation/structural_mutation.py` |
| §IV.B | LLM-guided boundary edge mutation | `src/augmentation/edge_mutation.py` |
| §IV.C | Three semantic-mutation strategies | `src/augmentation/semantic_mutation.py` |
| §IV.C | Four unified-verification checks | `src/augmentation/verifier.py` |
| §IV.D | 3-layer typed GIN | `src/detection/gin_encoder.py` |
| §IV.D | Per-layer GRU temporal encoder | `src/detection/temporal_encoder.py` |
| §IV.D | Hard-sample-weighted SupCon + training loop | `src/detection/contrastive_learning.py` |
| §IV.D | 2-layer MLP detection head | `src/detection/classifier.py` |
| §IV.E | Key causal-path extraction | `src/interpretation/attack_subgraph.py` |
| §IV.E | Action-triple → NL log enhancement | `src/interpretation/attack_sequence.py` |
| §IV.E | Sentence-BERT + Chroma technique mapping | `src/interpretation/semantic_matching.py` |
| §IV.E | LCS alignment vs sequence library | `src/interpretation/global_alignment.py` |
| §V.A | Released malicious-entity + technique labels | `data/annotated_labels/` |
| §V.A | ATT&CK knowledge base | `data/attack_knowledge/mitre_attack/` |
| Supp D | Hyperparameters T, r, L, D | `configs/athena.yaml` |
| Supp E | LLM configuration | `configs/athena.yaml::llm` |

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

`data/annotated_labels/` contains manually verified malicious-entity labels and ATT&CK technique labels for each malicious snapshot across DARPA E3, DARPA E5, OpTC, and ATLAS. ATLAS labels are released directly in this artifact:

- `data/annotated_labels/atlas/malicious_entities/M1-CVE-2015-5122_windows_h{1,2}.csv` — schema: `actorID, actor_type, objectID, object, action, timestamp`.
- `data/annotated_labels/atlas/attack_techniques/groundtruth.txt` — tab-separated ground-truth events.

The `darpa_e3`, `darpa_e5`, and `optc` directories contain placeholder structure; consult the supp page for the current release status of those labels.

## ATT&CK Knowledge Base (supp G.2 v)

- `data/attack_knowledge/mitre_attack/technique_triples_{raw,transformed}.json` — operation-level action triples for each ATT&CK technique, used by `src/interpretation/semantic_matching.py` as the retrieval corpus.
- `data/attack_knowledge/attackseqbench/technique_sequences.txt` — multi-stage attack-sequence library, used by `src/interpretation/global_alignment.py` for LCS alignment.

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
# Augmentation (paper §IV.B + §IV.C)
python scripts/run_augmentation.py   --config configs/athena.yaml --dataset cadets

# Detection (paper §IV.A + §IV.D)
python scripts/run_detection.py      --config configs/athena.yaml --dataset cadets

# Interpretation (paper §IV.E)
python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets
```

Supported `--dataset` values: `cadets, theia, trace, clearscope` (DARPA E3); `cadets5, theia5` (DARPA E5); `optcday1` (OpTC day 1); `atlas` (ATLAS).

Optional flags: `--scene <name>` to filter a specific scene (e.g. `cadets314`); `--epochs N` and `--max-snapshots N` to constrain runs.

## Configuration

Copy the local-secrets template and fill in your LLM API key:

```bash
cp local_settings.example.py local_settings.py
# Edit local_settings.py: CHATANYWHERE_API_KEY, CHATANYWHERE_ENDPOINT
```

`local_settings.py` is gitignored.

## Hyperparameters (supp D)

Defaults in `configs/athena.yaml` reproduce the values used in the paper:

| Symbol | Meaning | Default | Section |
|---|---|---|---|
| T | snapshot window | 1 min | snapshot |
| r | r-hop neighbourhood radius | 2 | gin |
| L | GIN layers | 3 | gin |
| D | embedding dimension | 128 | gin |
| top_k | WL retrieval candidates | 5 | augmentation |
| top_m | accepted mutations per anchor | 3 | augmentation |
| δ_h | WL similarity range (hardness check) | [0.30, 0.95] | augmentation |
| γ | mapping confidence cutoff | 0.40 | interpretation |
| LCS ratio | sequence-alignment cutoff | 0.60 | interpretation |

## Reproducibility Notes

- Seeds are set in `src/utils/seed.py` (call before training).
- LLM stochasticity is bounded by `temperature` in `configs/athena.yaml::llm`; runs against hosted LLMs are not bit-exact reproducible.

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
