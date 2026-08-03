"""Paper §IV.E — Global sequence alignment via longest common subsequence.

A predicted ATT&CK technique sequence is matched against the curated
attack-sequence library at
``data/attack_knowledge/attackseqbench/technique_sequences.txt`` using
longest common subsequence (LCS). The released implementation follows the
paper criterion ``LCS(Q, R) / min(|Q|, |R|)`` and admits sequences whose score
exceeds ``lcs_min_ratio`` in ``configs/athena.yaml::interpretation``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple


def load_technique_sequence_library(path: Optional[str] = None) -> List[List[str]]:
    """Load attack-technique sequences from ``path``. Lines starting with ``#``
    or that are empty are skipped. Within a line, tokens may be separated by
    whitespace or commas."""
    if path is None:
        path = str(Path(__file__).resolve().parents[2]
                   / "data" / "attack_knowledge" / "attackseqbench"
                   / "technique_sequences.txt")
    library: List[List[str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                tokens = [tok for chunk in line.split() for tok in chunk.split(",") if tok]
                if tokens:
                    library.append(tokens)
    except FileNotFoundError:
        return []
    return library


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
    """Return the library sequence with the highest paper LCS/min score."""
    best_seq: Optional[List[str]] = None
    best_score: float = 0.0
    for ref in library:
        if not ref:
            continue
        score = lcs_min_ratio(predicted, ref)
        if score > best_score:
            best_score = score
            best_seq = ref
    if best_score >= min_ratio:
        return best_seq, best_score
    return None, best_score


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
