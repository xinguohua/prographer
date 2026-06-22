# ATLAS — annotated labels

This directory ships the manually verified labels used in paper §V (ATLAS scene
M1-CVE-2015-5122).

## Layout

- `malicious_entities/M1-CVE-2015-5122_windows_h{1,2}.csv` — per-host event
  table. The two files cover host 1 (initial compromise + persistence) and host
  2 (lateral movement and exfiltration).
- `attack_techniques/groundtruth.txt` — tab-separated ground-truth event log
  used as the gold reference when scoring technique mapping.

## Schema

The CSVs are comma-separated, with header on line 1:

| Column      | Meaning                                       |
|-------------|-----------------------------------------------|
| `actorID`   | subject node identifier (process / user UUID) |
| `actor_type`| subject entity type (e.g. `PRINCIPAL_LOCAL`)  |
| `objectID`  | object node identifier                        |
| `object`    | object entity type (`FILE_OBJECT_BLOCK`, etc) |
| `action`    | edge operation (`executed`, `fork`, ...)      |
| `timestamp` | event timestamp (dataset native units)        |

`groundtruth.txt` follows the same six-column schema, tab-delimited.

## Provenance

Annotations were produced by three doctoral researchers in system security
based on the official ATLAS attack report; disagreements were resolved by
discussion (see paper §V.A and supp G.1).

## License

Released under the same redistribution terms as the upstream ATLAS dataset
(<https://github.com/purseclab/ATLAS>). Use these labels with the corresponding
ATLAS raw audit traces.
