# DARPA E3 — annotated labels

Placeholder. Released label files for DARPA E3 (Trace, Theia, Cadets,
ClearScope) will be published alongside the paper.

## Expected layout (matches ATLAS)

- `malicious_entities/` — per-scene CSV with the six-column schema
  `actorID, actor_type, objectID, object, action, timestamp`.
- `attack_techniques/` — per-scene ground-truth files (TSV) and ATT&CK
  technique mappings.

## Provenance

Labels are produced by three doctoral researchers in system security based on
the published DARPA E3 attack reports; disagreements are resolved by
discussion. See paper §V.A and supp G.1.

## Upstream dataset

<https://github.com/darpa-i2o/Transparent-Computing>
