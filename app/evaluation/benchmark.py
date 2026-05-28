"""
Run the full evaluation benchmark.

Usage:
    python -m app.evaluation.benchmark

Or via Makefile:
    make eval
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from app.config.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    from app.evaluation.ragas_eval import run_ragas_eval
    from app.evaluation.hallucination_check import run_hallucination_check
    import json

    print("=" * 60)
    print("MediGraph-RAG Evaluation Benchmark")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Step 1 — Run RAGAS eval
    print("\nRunning RAGAS evaluation...")
    scores = await run_ragas_eval()

    print("\nRESULTS:")
    print("-" * 40)

    if "error" in scores and "faithfulness" not in scores:
        print(f"RAGAS scoring error: {scores['error']}")
        print("Raw results still collected — check logs")
    else:
        print(f"Faithfulness:      {scores.get('faithfulness', 'N/A')}")
        print(f"Answer Relevancy:  {scores.get('answer_relevancy', 'N/A')}")

    print(f"Questions evaluated: {scores.get('questions_evaluated', 0)}")

    # Step 2 — Run hallucination check on each answer
    print("\nRunning hallucination checks...")
    print("-" * 40)

    per_question = scores.get("per_question", [])
    passed = 0
    failed = 0

    for item in per_question:
        context = " ".join(item.get("contexts", []))
        answer = item.get("answer", "")
        question = item.get("question", "")

        result = run_hallucination_check(
            reply=answer,
            context=context,
        )

        status = "PASS" if result["passed"] else "FAIL"
        if result["passed"]:
            passed += 1
        else:
            failed += 1

        print(f"[{status}] {question[:60]}")
        if not result["passed"]:
            print(
                f"       Unverified dosages: {result['unverified_dosages']}"
            )
            print(
                f"       Suspicious numbers: {result['suspicious_numbers']}"
            )

    print("-" * 40)
    print(f"Hallucination check: {passed} passed, {failed} failed")

    # Step 3 — Save results to file
    output = {
        "timestamp": datetime.now().isoformat(),
        "ragas_scores": {
            "faithfulness":     scores.get("faithfulness"),
            "answer_relevancy": scores.get("answer_relevancy"),
        },
        "hallucination_check": {
            "passed": passed,
            "failed": failed,
            "total":  len(per_question),
        },
        "per_question": per_question,
    }

    results_path = Path("datasets/processed/eval_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nFull results saved to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())