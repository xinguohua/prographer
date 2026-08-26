"""Paper §IV.E - Global sequence alignment via longest common subsequence.

A predicted ATT&CK technique sequence is matched against the deterministic,
source-linked AttackSeqBench derivative at
``data/attack_knowledge/attackseqbench/verified_sequences.jsonl`` using
longest common subsequence (LCS). The consistency score follows Eq. (8):
``LCS(Q, R) / min(|Q|, |R|)``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:[./]\d{3})?$", re.IGNORECASE)


def load_technique_sequence_records(path: Optional[str]) -> List[Dict]:
    """Load provenance-bearing AttackSeqBench-derived JSONL records.

    Every row must contain a stable ``source_id``, a non-empty
    ``source_record`` locator, and an ordered ``techniques`` list.  The legacy
    unlabeled text list is deliberately not accepted as scientific evidence.
    """
    if not path:
        return []
    source = Path(path)
    if not source.exists():
        return []
    records: List[Dict] = []
    seen = set()
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid attack-sequence JSONL at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"attack-sequence line {line_number} must be an object")
            source_id = str(row.get("source_id") or "").strip()
            source_record = str(row.get("source_record") or "").strip()
            source_hash = str(row.get("source_hash") or "").strip().lower()
            source_corpus = str(row.get("source_corpus") or "").strip()
            techniques = row.get("techniques")
            if (
                not source_id or not source_record or not source_corpus
                or not re.fullmatch(r"[0-9a-f]{64}", source_hash)
                or not isinstance(techniques, list) or not techniques
            ):
                raise ValueError(
                    f"attack-sequence line {line_number} requires source_id, source_record, "
                    "source_hash, source_corpus, techniques"
                )
            normalized = [str(value).strip().upper().replace("/", ".") for value in techniques]
            if any(not _TECHNIQUE_RE.fullmatch(value) for value in normalized):
                raise ValueError(f"invalid ATT&CK technique at line {line_number}")
            if source_id in seen:
                raise ValueError(f"duplicate attack-sequence source_id: {source_id}")
            seen.add(source_id)
            records.append({
                **row,
                "source_id": source_id,
                "source_record": source_record,
                "source_hash": source_hash,
                "source_corpus": source_corpus,
                "techniques": normalized,
            })
    return records


def load_technique_sequence_library(path: Optional[str] = None) -> List[List[str]]:
    return [row["techniques"] for row in load_technique_sequence_records(path)]


def lcs_length(a: List[str], b: List[str]) -> int:
    """Standard quadratic-time LCS length."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    if m < n:
        a, b = b, a
        m, n = n, m
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[n]


def lcs_min_ratio(predicted: List[str], reference: List[str]) -> float:
    """Return the paper's LCS consistency score.

    The denominator follows Eq. (8): ``min(|Q|, |R_j|)``.
    """
    if not predicted or not reference:
        return 0.0
    return lcs_length(predicted, reference) / min(len(predicted), len(reference))


def lcs_full_match_score(predicted: List[str], reference: List[str]) -> float:
    """Return the Eq. (8) LCS/min consistency score for chain filtering."""
    return lcs_min_ratio(predicted, reference)


def lcs_indices_keep_mask(a: List[str], b: List[str]) -> Tuple[List[bool], int]:
    """Return a boolean mask over ``a`` marking positions that participate in
    one specific LCS of ``a`` and ``b``, together with the LCS length."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return [False] * m, 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = dp[i - 1][j] if dp[i - 1][j] >= dp[i][j - 1] else dp[i][j - 1]
    keep = [False] * m
    i, j = m, n
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            keep[i - 1] = True
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return keep, dp[m][n]


def best_library_match(
    predicted: List[str],
    library: List[List[str]],
    min_ratio: float = 0.60,
) -> Tuple[Optional[List[str]], float]:
    """Return the best library match under the paper's LCS/min criterion."""
    best_seq: Optional[List[str]] = None
    best_score: float = 0.0
    for ref in library:
        if not ref:
            continue
        score = lcs_full_match_score(predicted, ref)
        if score > best_score:
            best_score = score
            best_seq = ref
    if best_score >= min_ratio:
        return best_seq, best_score
    return None, best_score


def build_candidate_tactic_chains(
    queue_entries: Sequence[Dict],
    top_k: int,
) -> List[Dict]:
    """Beam-compose the persistent queue's top-K tactic candidate sets.

    Each queue entry contains ``candidates=[{tactic, score, ...}]``.  Adjacent
    duplicate tactics are collapsed, and only the globally best ``top_k``
    distinct chains are retained after every queue item.
    """
    beam: List[Tuple[List[str], float]] = [([], 0.0)]
    width = max(1, int(top_k))
    for entry in queue_entries:
        candidates = entry.get("candidates", []) if isinstance(entry, dict) else []
        valid = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            tactics = candidate.get("tactics")
            if not isinstance(tactics, list):
                tactics = [candidate.get("tactic", "")]
            for tactic in tactics:
                tactic = str(tactic).strip()
                if tactic:
                    valid.append({**candidate, "tactic": tactic})
        if not valid:
            continue
        composed: Dict[Tuple[str, ...], float] = {}
        for prefix, prefix_score in beam:
            for candidate in valid:
                tactic = str(candidate["tactic"]).strip()
                chain = list(prefix)
                if not chain or chain[-1] != tactic:
                    chain.append(tactic)
                score = prefix_score + float(candidate.get("score", 0.0))
                key = tuple(chain)
                composed[key] = max(composed.get(key, float("-inf")), score)
        beam = [
            (list(chain), score)
            for chain, score in sorted(composed.items(), key=lambda item: (-item[1], item[0]))[:width]
        ]
    return [
        {"tactics": chain, "semantic_score": score}
        for chain, score in beam if chain
    ]


def align_candidate_chains(
    candidate_chains: Sequence[Dict],
    library: Sequence[Sequence[str]],
    min_ratio: float,
    top_k: int,
) -> List[Dict]:
    """Align every candidate chain and rank the top-K LCS/min matches."""
    aligned: List[Dict] = []
    for candidate in candidate_chains:
        chain = list(candidate.get("tactics", []))
        best_ref: Optional[List[str]] = None
        best_score = 0.0
        best_source = None
        for reference_row in library:
            if isinstance(reference_row, dict):
                reference = list(reference_row.get("tactics", []))
                source = {
                    key: reference_row.get(key) for key in (
                        "source_id", "source_record", "source_hash", "source_corpus",
                    )
                }
            else:
                reference = list(reference_row)
                source = None
            if not reference:
                continue
            score = lcs_full_match_score(chain, reference)
            if score > best_score:
                best_ref, best_score, best_source = reference, score, source
        aligned.append({
            "tactics": chain,
            "semantic_score": float(candidate.get("semantic_score", 0.0)),
            "library_match": best_ref if best_score >= min_ratio else None,
            "library_source": best_source if best_score >= min_ratio else None,
            "full_match_score": best_score,
            "passes_threshold": best_score >= min_ratio,
        })
    aligned.sort(key=lambda row: (-row["full_match_score"], -row["semantic_score"], row["tactics"]))
    return aligned[:max(1, int(top_k))]


def filter_positive_by_tech_lcs(
    predicted_per_snapshot: List[List[str]],
    library: List[List[str]],
    min_ratio: float = 0.60,
) -> List[bool]:
    """Per-snapshot: keep a positive prediction only if its predicted technique
    sequence has an LCS/min match in the library above ``min_ratio``. Returns a
    boolean mask parallel to ``predicted_per_snapshot``."""
    keep: List[bool] = []
    for predicted in predicted_per_snapshot:
        _seq, ratio = best_library_match(predicted, library, min_ratio=min_ratio)
        keep.append(ratio >= min_ratio)
    return keep
