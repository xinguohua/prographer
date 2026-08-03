from src.interpretation.global_alignment import (
    best_library_match,
    lcs_full_match_score,
    lcs_min_ratio,
)


def test_lcs_min_ratio_uses_paper_denominator():
    reference = ["Initial Access", "Persistence", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Execution", "Exfiltration"]

    assert lcs_min_ratio(predicted, reference) == 1.0


def test_full_match_uses_paper_lcs_min_score():
    reference = ["Initial Access", "Persistence", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Execution", "Exfiltration"]

    assert lcs_full_match_score(predicted, reference) == 1.0


def test_single_tactic_uses_eq8_without_extra_stage_gate():
    reference = ["Initial Access", "Persistence", "Execution", "Exfiltration"]
    predicted = ["Execution"]

    assert lcs_min_ratio(predicted, reference) == 1.0
    assert best_library_match(predicted, [reference], min_ratio=0.6) == (reference, 1.0)


def test_lcs_min_ratio_partial_order_match():
    reference = ["Initial Access", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Persistence", "Collection"]

    assert round(lcs_min_ratio(predicted, reference), 4) == 0.3333
