import json
import logging
import asyncio
from pathlib import Path
from app.agents.graph_builder import run_rag_graph

logger = logging.getLogger(__name__)


async def run_single_eval(
    question: str,
    ground_truth: str,
    patient_id: str,
) -> dict:
    """
    Run one question through the RAG pipeline and
    collect the answer and context for scoring.

    Returns:
        Dict with question, answer, contexts, ground_truth
        (this is the format RAGAS expects)
    """
    logger.info(f"Evaluating: '{question[:60]}'")

    try:
        result = await run_rag_graph(
            message=question,
            patient_id=patient_id,
            session_id="eval-session",
        )

        # Extract context texts from sources
        contexts = [
            s.get("text", "")
            for s in result.get("sources", [])
        ]

        # If no sources (e.g. no chart uploaded), use context string
        if not contexts and result.get("context"):
            contexts = [result["context"]]

        return {
            "question":     question,
            "answer":       result.get("reply", ""),
            "contexts":     contexts,
            "ground_truth": ground_truth,
        }

    except Exception as e:
        logger.error(f"Eval failed for question '{question}': {e}")
        return {
            "question":     question,
            "answer":       "",
            "contexts":     [],
            "ground_truth": ground_truth,
        }


async def run_ragas_eval(
    eval_path: str = "datasets/processed/eval_qa.json",
) -> dict:
    """
    Run the full RAGAS evaluation suite.

    Loads questions from eval_qa.json, runs each through
    the RAG pipeline, then scores with RAGAS metrics.

    Returns:
        Dict with metric scores and per-question results
    """
    path = Path(eval_path)
    if not path.exists():
        logger.error(f"Eval dataset not found at: {eval_path}")
        return {"error": f"File not found: {eval_path}"}

    with open(path, "r") as f:
        eval_data = json.load(f)

    logger.info(f"Running eval on {len(eval_data)} questions")

    # Run all questions through the pipeline
    results = []
    for item in eval_data:
        result = await run_single_eval(
            question=item["question"],
            ground_truth=item["ground_truth"],
            patient_id=item["patient_id"],
        )
        results.append(result)
        # Small delay to avoid overwhelming Ollama
        await asyncio.sleep(1)

    # Try RAGAS scoring
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset

        dataset = Dataset.from_list(results)

        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy],
        )

        logger.info(f"RAGAS scores: {scores}")

        return {
            "faithfulness":     round(float(scores["faithfulness"]), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
            "questions_evaluated": len(results),
            "per_question": results,
        }

    except Exception as e:
        logger.error(f"RAGAS scoring failed: {e}")

        # Return raw results even if RAGAS scoring fails
        return {
            "error": str(e),
            "note": "Raw results collected but RAGAS scoring failed",
            "questions_evaluated": len(results),
            "per_question": results,
        }