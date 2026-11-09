"""ROUGE scoring and the lead-sentences baseline."""
from rouge_score import rouge_scorer

METRICS = ("rouge1", "rouge2", "rougeL")
_scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)


def lead_baseline(sentences, count=2):
    return " ".join(sentences[:count])


def score(prediction, references):
    """F1 per metric, taking the best reference for each metric."""
    results = [_scorer.score(reference, prediction) for reference in references]
    return {m: max(result[m].fmeasure for result in results) for m in METRICS}


def mean(rows):
    return {m: sum(row[m] for row in rows) / len(rows) for m in METRICS}


def beats(model, baseline):
    return all(model[m] > baseline[m] for m in METRICS)
