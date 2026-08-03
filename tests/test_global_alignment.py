from src.interpretation.global_alignment import lcs_f1_score


def test_lcs_f1_penalizes_missing_reference_stage():
    reference = ["Initial Access", "Persistence", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Execution", "Exfiltration"]

    score, parts = lcs_f1_score(predicted, reference)

    assert parts["lcs"] == 3
    assert parts["coverage"] == 0.75
    assert parts["precision"] == 1.0
    assert round(score, 4) == 0.8571


def test_lcs_f1_penalizes_extra_predicted_stage():
    reference = ["Initial Access", "Execution", "Exfiltration"]
    predicted = ["Initial Access", "Persistence", "Execution", "Exfiltration"]

    score, parts = lcs_f1_score(predicted, reference)

    assert parts["lcs"] == 3
    assert parts["coverage"] == 1.0
    assert parts["precision"] == 0.75
    assert round(score, 4) == 0.8571
