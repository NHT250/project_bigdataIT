"""Recall@5 and recall@10 of each retrieval mode on rag/eval/retrieval_questions.json.

    PYTHONPATH=. python -m rag.eval_retrieval

Recall@k is the share of a question's answer articles found in the top k, averaged over the
questions. Each question here has one answer article, so it is the share of questions whose
article was found.
"""
import json
import statistics
import sys
import time
from pathlib import Path

QUESTIONS = Path(__file__).parent / "eval" / "retrieval_questions.json"
CUTOFFS = (5, 10)


def recall(found, gold, k):
    return len(set(found[:k]) & gold) / len(gold)


def rank_of(found, gold):
    ranks = [found.index(key) + 1 for key in gold if key in found]
    return min(ranks) if ranks else None


def evaluate(conn, embedder, questions, modes):
    from rag import retrieval

    results = {mode: [] for mode in modes}
    for question in questions:
        vector = embedder.encode([question["question"]])[0]
        gold = {(g["source"], g["source_id"]) for g in question["gold"]}
        for mode in modes:
            start = time.perf_counter()
            found = retrieval.ranked_articles(conn, question["question"], vector, mode, max(CUTOFFS))
            results[mode].append({
                "id": question["id"], "kind": question["kind"], "rank": rank_of(found, gold),
                "seconds": time.perf_counter() - start,
                **{f"recall@{k}": recall(found, gold, k) for k in CUTOFFS},
            })
    return results


def main():
    import psycopg

    from rag import retrieval, settings
    from rag.embedder import Embedder

    sys.stdout.reconfigure(encoding="utf-8")
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    with psycopg.connect(settings.database_url(), autocommit=True) as conn:
        results = evaluate(conn, Embedder(), questions, retrieval.MODES)

    print("Rank of the answer article (- = not in the top 10):")
    print("  id  kind        " + "  ".join(f"{mode:>6}" for mode in results))
    for index, question in enumerate(questions):
        ranks = [results[mode][index]["rank"] for mode in results]
        print(f"  {question['id']:>2}  {question['kind']:<10}  " + "  ".join(f"{r if r else '-':>6}" for r in ranks))
    print()
    for kind in (None, "name", "paraphrase"):
        label = "all 20" if kind is None else f"{kind} only"
        print(f"{label}:")
        for mode, rows in results.items():
            rows = [r for r in rows if kind is None or r["kind"] == kind]
            scores = "  ".join(f"recall@{k} {statistics.mean(r[f'recall@{k}'] for r in rows):.2f}" for k in CUTOFFS)
            print(f"  {mode:<6}  {scores}  (n={len(rows)}, median {statistics.median(r['seconds'] for r in rows) * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
