from src.interpretation.global_alignment import lcs_min_ratio


def test_lcs_min_ratio_uses_paper_denominator():
    reference = ["Initial Access", "Persistence", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Execution", "Exfiltration"]

    assert lcs_min_ratio(predicted, reference) == 1.0


def test_lcs_min_ratio_partial_order_match():
    reference = ["Initial Access", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Persistence", "Collection"]

    assert round(lcs_min_ratio(predicted, reference), 4) == 0.3333
