# ATHENA

ATHENA is a provenance-based intrusion detection system for stealthy Advanced Persistent Threats (APTs). It builds time-windowed provenance snapshots from system audit logs, learns discriminative node representations through LLM-guided adaptive contrastive learning, performs **per-node binary benign/malicious detection**, and reconstructs interpretable multi-stage attack chains by mapping each malicious node to an ATT&CK technique and **aligning the resulting tactic sequence** against multi-stage attack patterns.

The online supplementary material lives at <https://xinguohua.github.io/athena-supp/>.

## Method Overview

ATHENA has four stages:

1. **Snapshot construction.** Audit events are partitioned into 1-minute non-overlapping windows; each window is materialized as a typed provenance graph and decomposed into node-centred *r*-hop subgraphs. → `src/snapshot_construction/`.
2. **LLM-guided graph augmentation.** After the evaluation split is fixed, each training-only benign anchor retrieves training-only attack graphs via the Weisfeiler–Leman subtree kernel. The single best aligned region across the Top-N references is mutated; each injected process receives exactly one of Replacement, Rewriting, or Extension. A unified verification step filters mutations that violate operation legality, attribute feasibility, imperceptibility, or hardness. → `src/augmentation/`.
3. **Adaptive contrastive learning and node-level detection.** A 3-layer GIN produces an instantaneous representation, followed by one GRU update per node and snapshot. The final state of each benign *r*-hop graph's centre node is trained against centre-node states from the train-only original and verified synthetic attack corpus with the hard-sample-weighted contrastive loss. A 2-layer MLP head emits a benign/malicious label for each node. → `src/detection/`.
4. **Technique mapping and tactic-level alignment.** For every detector-flagged node, ATHENA reconstructs the causal paths to other flagged nodes (or uses the bounded λ-hop fallback), retrieves a top-K parent-technique/tactic set, appends it to the persistent queue, and LCS/min-aligns the resulting top-K candidate tactic chains against AttackSeqBench. → `src/interpretation/`.

## Repository Layout

```
.
├── configs/athena.yaml         # paths + hyperparameters
├── prompts/                    # LLM prompt templates for augmentation
├── data/
│   ├── annotated_labels/       # malicious-entity registry + available source-linked ATT&CK annotations
│   └── attack_knowledge/       # ATT&CK technique KB + tactic-sequence library
├── src/
│   ├── snapshot_construction/  # 1-min provenance snapshots + r-hop ego subgraphs
│   ├── augmentation/           # WL retrieval + structural / semantic / edge mutation + verifier
│   ├── detection/              # GIN + final-layer GRU, contrastive loss, node MLP head
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

Reference environment: Python 3.9, Ubuntu, and a CUDA-capable GPU.

## Data Preparation

Download the upstream datasets and unpack them locally:

| Dataset | Source |
|---|---|
| DARPA E3 (Trace, Theia, Cadets, ClearScope) | <https://github.com/darpa-i2o/Transparent-Computing> |
| DARPA E5 (Trace, Theia, Cadets, ClearScope) | <https://github.com/darpa-i2o/Transparent-Computing-5D> |
| DARPA OpTC | <https://github.com/FiveDirections/OpTC-data> |
| ATLAS | <https://github.com/purseclab/ATLAS> |

ATHENA consumes the following canonical **indexed-event representation**:

- E3/E5: `<DATASET_ROOT>/<scene>/{benign,malicious}/*.json` plus
  `<DATASET_ROOT>/<scene>_{benign,malicious}.txt`. Each tab-separated index row
  is `actorID, actor_type, objectID, object_type, action, timestamp` and must
  refer to an event in the corresponding JSON files.
- OpTC: for a JSON file `<P>/<a>/<b>/<file>.json`, provide the companion
  tab-separated edge index as `<P>/<a>_<b>_<file>.txt`, with the same six
  logical columns (the parser also accepts the released `Source_ID`,
  `Source_Type`, `Destination_ID`, `Destination_Type`, `Edge_Type`,
  `Timestamp` headers).
- ATLAS: point `paths.atlas.root` at the unpacked official ATLAS v1 release.
  The loader directly consumes `training_logs|testing_logs/<case>/logs` plus
  the same case's `malicious_labels.txt`, or the official
  `paper_experiments/output/{training,testing}_preprocessed_logs_*` files.
  The source timezone is explicit in `paths.atlas.source_timezone`. An indexed
  export remains supported as `<ATLAS_ROOT>/<fold>/<host>_events.csv`, but it
  must include `timestamp_unit` (`s`, `ms`, `us`, or `ns`) and the official
  scenario `malicious_labels.txt`; command, arguments, path, and address
  columns are retained when present.

Dataset paths in `configs/athena.yaml` point to the configured dataset roots.
The parsers consume their documented audit-log or event-table inputs and
construct node properties from the event rows in each snapshot.

## Released Labels (supp G.1)

`data/annotated_labels/` provides malicious-entity labels for snapshot construction,
supervised training, and detector metrics. Technique/tactic evaluation consumes
separate source-linked annotation records through the documented schema.

- `<dataset>/malicious_entities/` — malicious-entity labels at the scope of the corresponding public source: E3 scene, E5 platform/attack registry, or OpTC host. UUID text files contain one provenance-node identifier per line. The shared registry supplies snapshot construction and supervised evaluation; the OpTC paper profile uses `host_0051`, `host_0201`, and `host_0501` (H051/H201/H501). ATLAS v1 is different: its loader consumes the semantic labels shipped beside each official case and applies the official endpoint-wise substring projection.
- `<dataset>/attack_techniques/` — normalized parent-level MITRE ATT&CK annotations. The E5, OpTC, and ATLAS v1 bundles contain exact source-linked mappings; the four E3 JSON files contain 49 scene/entity annotations without exact audit-event boundaries.

The released exact mapping bundles contain 43 records:

| Bundle | Exact mappings | Scope |
|---|---:|---|
| DARPA E5 | 6 | Four exact attack scenes across Cadets, ClearScope, Theia, and Trace; four additional E5 scenes remain unassigned at exact-event granularity |
| OpTC | 27 | H051/day 3, H201/day 1, and H501/day 2 |
| ATLAS v1 | 10 | Six PL and four PA scenario mappings |

Each exact mapping binds a reviewed ATT&CK technique/tactic to its source
record, host, graph event ID, raw audit-event ID, anchor and role, timestamp,
snapshot coordinate, and content hashes. The accompanying
`source_records.jsonl`, `source_linked_annotations.jsonl`,
`mapping_records.jsonl`, and `content_manifest.json` files preserve the joins.
These 43 released mappings are distinct from the paper-profile RQ3 inputs:
53 mapping records, 53 E3 sequence records, 53 E5 sequence records, their
attack-event boundaries, and the 24/48/72-hour benign-source plans. The RQ3
evaluator and replay builder implement and validate those input contracts.

The following command imports public/local ground-truth sources into the
normalized malicious-entity label schema:

```bash
python scripts/import_ground_truth_labels.py \
  --pidsmaker-root "$PIDSMaker_ROOT"
```

The importer is pinned to PIDSMaker commit
`32602734bc9f896be5fc0f03f0a185c967cd6624`. It reads the first field of each
OrTHRUS CSV as the malicious provenance-node UUID, preserves the node
attributes and PIDSMaker-local index in a source manifest, and writes the
paper-profile OpTC labels separately for H051/H201/H501. Audit-event IDs are
resolved from the raw event records and are never mixed into the node-label
files. ATLAS v1 labels are read from the official case directories; PIDSMaker's
ATLASv2 labels are a different dataset and are not used for the ATLAS results.

## ATT&CK Knowledge Base (supp G.2 v)

- `data/attack_knowledge/mitre_attack/technique_triples_{raw,transformed}.json` — operation-level action triples for each ATT&CK technique, used by `src/interpretation/semantic_matching.py` as the retrieval corpus.
- `interpretation.attack_sequence_records` points to a source-linked JSONL
  record set used by LCS/min alignment. Each row requires
  `source_id`, `source_record`, `source_hash`, `source_corpus`, and an ordered
  `techniques` list; validate and normalize it with
  `scripts/import_attack_sequence_records.py`.

The default 408-record derivative is deterministically built from the official
[AttackSeqBench source](https://anonymous.4open.science/r/AttackSeqBench)
([arXiv:2503.03170](https://arxiv.org/abs/2503.03170)). It retains only ordered
ATT&amp;CK technique/tactic identifiers, record titles, relative source locators,
and hashes—not CTI report text. The stable <code>content_manifest.json</code>
binds every source locator, raw-record hash, canonical derived record, and the
aggregate derived corpus; the ZIP checksum is retained separately as metadata
for the particular retrieval. Rebuild it from an official source download with:

```bash
python scripts/import_attack_sequence_records.py \
  --raw-root "$ATTACKSEQ_ROOT" \
  --source-archive "$ATTACKSEQ_ARCHIVE" \
  --expected-records 408
```

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
python scripts/run_augmentation.py   --config configs/athena.yaml --dataset cadets \
  --model gpt-4o

# Detection (paper §IV.A + §IV.D): consumes the admitted augmentations
# produced by the preceding stage,
# then writes held-out predictions and metrics
python scripts/run_detection.py      --config configs/athena.yaml --dataset cadets \
  --mode complete \
  --augmented-dir outputs/augmented_graphs \
  --execution train-save --checkpoint-out outputs/cadets_basic.pt \
  --output outputs/detection_predictions.json

# Interpretation (paper §IV.E): maps detector positives to ATT&CK techniques
# and tactic sequences; this base command does not score Table VII/VIII
python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets \
  --detections outputs/detection_predictions.json \
  --output outputs/interpretation/full-enhanced.json

# Artifact validation (the MMD check also requires a manifest generated from
# the real reference and variant embedding arrays for that run)
python scripts/validate_artifact.py

# Human-rating agreement check for the released rating sheet
python scripts/compute_rating_agreement.py --ratings data/human_ratings.csv
```

`--mode complete` requires a split-matched manifest with at least one admitted
variant. The paper-consistent no-augmentation ablation is run explicitly with
`--mode ablation-no-augmentation`; its output is labelled as that variant.
`scripts/compute_mmd.py` accepts real reference/variant embedding arrays and
writes a seeded permutation manifest with input hashes. A concrete MMD run
supplies the reference and variant arrays emitted by its trained GIN encoder;
the manifest binds those run-specific arrays to the reported statistics.

RQ3 Table VII top-1/3/5 Acc/STE/CTE/Unmapped scoring and Table VIII
Basic/24h/48h/72h/E5 LCS-based FM/PM/Miss scoring are implemented by
`scripts/evaluate_interpretation.py` for provenance-bearing source-linked GT JSONL.
Table VII runs `run_interpretation.py` once with each
`--mapping-variant {direct,tech-enhanced,log-enhanced,full-enhanced}`; these
switch the raw/transformed technique triples and raw/abstracted log mapping,
respectively. Table VIII reuses the Basic E3 checkpoint without retraining.
The table-specific RQ3 registry contract requires 53 `mapping` records, 53 E3
`sequence` records, 53 E5 `sequence` records, and their corresponding
`attack_event` boundary records. Every scored row binds its dataset, source
scene, stable host, anchor, audit `event_id`, snapshot, and event hash. Each table-specific benign
source plan binds consecutive E3 attack-event gaps to complete train-only
benign audit windows, snapshot hashes, and an explicit reuse policy. These
table-specific records and 24/48/72-hour plans are separate from the
43 exact mapping records released under `data/annotated_labels/`.

The following is an interface example for one Cadets condition. Set
`RQ3_REGISTRY` and `CADETS_24H_PLAN` to table-specific inputs satisfying the
contracts above; the full Table VII/VIII evaluation additionally uses all four
mapping branches, the corresponding E3/E5 sequence rows, and all three replay
conditions.

```bash
python scripts/build_benign_injection_manifest.py --config configs/athena.yaml \
  --dataset cadets --condition 24h \
  --attack-event-boundaries "$RQ3_REGISTRY" \
  --benign-source-plan "$CADETS_24H_PLAN" \
  --checkpoint outputs/cadets_basic.pt \
  --output outputs/replay/24h/events.json \
  --spec-output outputs/replay/24h/spec.json
python scripts/run_interval_replay.py --spec outputs/replay/24h/spec.json \
  --output-dir outputs/replay/24h/run

# E3-to-E5 unseen evaluation: frozen E3 encoder/GRU/Word2Vec/MLP, no E5 fitting
python scripts/run_detection.py --config configs/athena.yaml --dataset cadets5 \
  --execution eval-only --checkpoint outputs/cadets_basic.pt \
  --output outputs/cadets5_transfer.json
python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets5 \
  --mapping-variant full-enhanced --detections outputs/cadets5_transfer.json \
  --attack-event-boundaries data/annotated_labels/darpa_e5/attack_techniques/mapping_records.jsonl \
  --output outputs/interpretation/cadets5-unseen.json

# Table VII interface: repeat interpretation for all four mapping branches
for variant in direct tech-enhanced log-enhanced full-enhanced; do
  python scripts/run_interpretation.py --config configs/athena.yaml --dataset cadets \
    --mapping-variant "$variant" --detections outputs/detection_predictions.json \
    --attack-event-boundaries "$RQ3_REGISTRY" \
    --output "outputs/interpretation/${variant}.json"
done

# Tables VII/VIII interface: score a complete table-specific registry and runs
python scripts/evaluate_interpretation.py \
  --interpretation outputs/interpretation/direct.json \
  --interpretation outputs/interpretation/tech-enhanced.json \
  --interpretation outputs/interpretation/log-enhanced.json \
  --interpretation outputs/interpretation/full-enhanced.json \
  --interpretation outputs/interpretation/cadets5-unseen.json \
  --replay-manifest outputs/replay/24h/run/manifest.json \
  --replay-manifest outputs/replay/48h/run/manifest.json \
  --replay-manifest outputs/replay/72h/run/manifest.json \
  --ground-truth "$RQ3_REGISTRY" \
  --attackseq-records data/attack_knowledge/attackseqbench/verified_sequences.jsonl \
  --output outputs/rq3_tables.json
```

The replay spec binds `dataset`, optional `scene`, `condition`, `config`, the
Basic E3 `checkpoint`, and the generated `source_event_manifest`. The resulting
run manifest records and hashes the consumed injection, checkpoint, detection,
and interpretation artifacts.

The date-partition command-line profiles use `cadets, theia, trace, clearscope` (DARPA E3), `cadets5, theia5, trace5, clearscope5` (DARPA E5), and `optcday1` (OpTC, using H051/H201/H501). Paper-protocol runs omit `--scene` so that all matching days are loaded before the attack-day split.

ATLAS uses the original scenario-level protocol. Select one held-out fold with
`--dataset atlas --scene <fold>`: `S1`–`S4` train on the other three
single-host scenarios, while `M1`–`M6` train on the other five multi-host
scenarios. Use a fold-specific augmentation directory and output file:

```bash
python scripts/run_augmentation.py --config configs/athena.yaml \
  --dataset atlas --scene S1 --model gpt-4o \
  --output-dir outputs/atlas/S1/augmented_graphs
python scripts/run_detection.py --config configs/athena.yaml \
  --dataset atlas --scene S1 \
  --augmented-dir outputs/atlas/S1/augmented_graphs \
  --output outputs/atlas/S1/detection_predictions.json
python scripts/run_interpretation.py --config configs/athena.yaml \
  --dataset atlas --scene S1 \
  --detections outputs/atlas/S1/detection_predictions.json \
  --attack-event-boundaries data/annotated_labels/atlas/attack_techniques/mapping_records.jsonl \
  --output outputs/atlas/S1/interpretation.json
```

Optional flags: `--scene <name>` filters a specific scene; `--model <key>` selects an LLM configuration; `--epochs N` and `--max-snapshots N` constrain test runs. Augmentation and detection use the same date-partition contract: all benign-only days and the earlier attack days are training data, while the remaining attack days are held out. Augmentation uses training benign anchors and training attack donors, and its manifest binds the dataset, scene, and split used by detection. The interpretation script consumes held-out detector positives.

## LLM Routing and Accounting

Each model row in `configs/athena.yaml` declares its provider, OpenAI-compatible
`base_url`, served model identifier, and API-key environment variable. For the
primary GPT-4o run:

```bash
export OPENAI_API_KEY='...'
python scripts/run_augmentation.py --config configs/athena.yaml \
  --dataset cadets --model gpt-4o
```

The hosted DeepSeek-V3-0324 alternative requires an explicitly pinned compatible
endpoint and served model identity; this avoids resolving a rolling provider alias:

```bash
export DEEPSEEK_API_KEY='...'
export DEEPSEEK_V3_BASE_URL='https://<v3-0324-compatible-service>/v1'
export DEEPSEEK_V3_MODEL_ID='<served-v3-0324-model-id>'
python scripts/run_augmentation.py --config configs/athena.yaml \
  --dataset cadets --model deepseek-v3
```

Local vLLM models
accept the default `EMPTY` key (or `VLLM_API_KEY` when the server requires one).
Run one local model at a time on the RTX 4090:

```bash
# Qwen2.5-7B, BF16
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --dtype bfloat16 \
  --max-model-len 4096 --gpu-memory-utilization 0.92 --port 8000

# Qwen2.5-14B, official AWQ/INT4 checkpoint (separate run)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-14B-Instruct-AWQ --quantization awq \
  --max-model-len 4096 --gpu-memory-utilization 0.92 --port 8000
```

Every edge- and semantic-mutation request is written to `manifest.json` with
stage, mutation attempt, provider/model, provider token usage, wall latency,
and API retry count. The manifest also contains stage totals and token cost.
Costs can be recomputed after a pricing update:

```bash
python scripts/recalculate_llm_cost.py \
  --manifest outputs/augmented_graphs/manifest.json \
  --config configs/athena.yaml
```

## Hyperparameters (supp D)

Defaults in `configs/athena.yaml` match the artifact configuration used by the released scripts:

| Symbol | Meaning | Default | Section |
|---|---|---|---|
| T | snapshot window | 1 min | snapshot |
| r | r-hop neighbourhood radius | 4 | gin |
| L | GIN layers | 3 | gin |
| D | embedding dimension | 64 | gin |
| β | HCL focusing intensity | 2.0 | contrastive |
| top_k | WL retrieval candidates | 5 | augmentation |
| retry budget | unified-verification retry limit | 3 | augmentation |
| δ_h | minimum WL similarity (hardness check) | 0.30 | augmentation |
| γ | mapping confidence cutoff | 0.50 | interpretation |
| K | technique/tactic candidate chains | 5 | interpretation |
| temporal | GRU temporal encoder | enabled | detection |
| epochs | detector training epochs | 3 | detection |
| train ratio | attack-day detector training split | 0.70 | detection |
| T_max | tactic queue retention window | 7 days | interpretation |
| LCS/min | sequence-alignment cutoff | 0.60 | interpretation |

## Citation

```bibtex
@misc{athena2026,
  title  = {Interpretable Stealthy APT Detection via LLM-Augmented Graph Contrastive Learning and Semantic Provenance Abstraction},
  author = {Guohua Xin and Guangquan Xu and Jiongchi Yu and Yao Zhang and Xiaofei Xie and Tuoyu Chen and Lingxiao Jiang and Pan Gao},
  year   = {2026},
  note   = {Manuscript submitted to IEEE Transactions on Information Forensics and Security; artifact: https://github.com/xinguohua/athena}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Released annotation labels follow the redistribution terms of the upstream datasets they annotate.
