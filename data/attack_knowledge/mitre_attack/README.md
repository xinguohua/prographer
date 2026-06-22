# MITRE ATT&CK — technique triples knowledge base

Used by `src/interpretation/semantic_matching.py` (paper §IV.E,
supp G.2 (v) — "ATT&CK technique knowledge base with action-triplet
enhancement").

## Files

- `technique_triples_raw.json` — raw action triples extracted from each ATT&CK
  technique description. Shape:
  ```
  {
    "T1059": [
      {"src_type": "process", "op": "execute", "dst_type": "process", ...},
      ...
    ],
    "T1078": [...]
  }
  ```
- `technique_triples_transformed.json` — operation-normalized triples ready for
  retrieval. Same outer shape as the raw file; loaded at runtime by the
  Sentence-BERT mapper.

## Provenance

Action triples are extracted from the MITRE ATT&CK technique descriptions
(<https://attack.mitre.org/>); the transformation script is part of the
ATHENA codebase. The triples are kept in JSON form so the retrieval index can
be rebuilt offline without re-running extraction.
