import pytest

import scoring

SENTENCES = ["We propose a parser.", "It is fast.", "It is also accurate on noisy text."]


def test_baseline_is_the_first_two_sentences():
    assert scoring.lead_baseline(SENTENCES) == "We propose a parser. It is fast."


def test_identical_text_scores_one_on_every_metric():
    result = scoring.score("a fast neural parser", ["a fast neural parser"])
    assert result == pytest.approx({"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0})


def test_unrelated_text_scores_zero():
    result = scoring.score("bananas ripen slowly", ["a fast neural parser"])
    assert result == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_best_reference_is_used_for_each_metric():
    single = scoring.score("a fast neural parser", ["a fast neural parser"])
    multi = scoring.score(
        "a fast neural parser", ["bananas ripen slowly", "a fast neural parser"]
    )
    assert multi == single


def test_mean_averages_each_metric():
    rows = [
        {"rouge1": 1.0, "rouge2": 0.5, "rougeL": 0.0},
        {"rouge1": 0.0, "rouge2": 0.5, "rougeL": 1.0},
    ]
    assert scoring.mean(rows) == {"rouge1": 0.5, "rouge2": 0.5, "rougeL": 0.5}


def test_model_must_beat_baseline_on_every_metric():
    baseline = {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.3}
    assert scoring.beats({"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4}, baseline)
    assert not scoring.beats({"rouge1": 0.5, "rouge2": 0.1, "rougeL": 0.4}, baseline)


def test_a_tie_does_not_count_as_beating_the_baseline():
    baseline = {"rouge1": 0.4, "rouge2": 0.2, "rougeL": 0.3}
    assert not scoring.beats(dict(baseline), baseline)
