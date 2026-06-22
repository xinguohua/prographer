# AttackSeqBench — attack-technique sequence library

Used by `src/interpretation/global_alignment.py` to score predicted technique
sequences via LCS against curated multi-stage attack patterns (paper §IV.E).

## File

`technique_sequences.txt` — one sequence per line, whitespace or comma
delimited. Lines starting with `#` are comments. Example:

```
T1547/004,T1078
T1547 T1055 T1027
T1059,T1053,T1105
```

Each token is a MITRE ATT&CK technique identifier; the order is the observed
kill-chain order. See `src/interpretation/global_alignment.py` for the matching
rule (LCS ratio threshold configurable via `configs/athena.yaml::interpretation.lcs_min_ratio`).
