"""
Experiment Evaluator

Handles execution of experiments and metric computation for each technique.
"""

from typing import Dict, List, Any, Optional
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_dataset
from llm import LLMClient, LLMResponse
from metrics import (
    calculate_dataset_accuracy,
    calculate_entropy,
    calculate_perplexity,
    calculate_loss,
    calculate_fallback_entropy,
    calculate_fallback_perplexity,
    calculate_fallback_loss,
    calculate_accuracy,
)
from prompts import PromptTemplate


def evaluate_technique(
    llm_client: LLMClient,
    dataset: Dict[str, Any],
    prompt_template: PromptTemplate,
    technique_name: str,
) -> Dict[str, Any]:
    """
    Evaluate a single technique on a dataset.

    Args:
        llm_client: LLM client for generation
        dataset: Dataset to evaluate on
        prompt_template: Prompt template for the technique
        technique_name: Name of the technique

    Returns:
        Dictionary with evaluation results
    """
    samples = dataset.get("samples", [])
    predictions = []
    responses = []
    has_logprobs = False

    print(f"  → Evaluating {technique_name} on {len(samples)} samples...")

    for i, sample in enumerate(samples):
        # Build prompt
        question = sample.get("question") or sample.get("problem", "")
        prompt_text = prompt_template.format(question=question)
        system_prompt = prompt_template.system_prompt

        # Generate response
        try:
            response = llm_client.generate(
                prompt=prompt_text,
                system_prompt=system_prompt,
                logprobs=True,
            )
            responses.append(response)

            # Check if logprobs are available
            if response.logprobs is not None and len(response.logprobs) > 0:
                has_logprobs = True

            # Extract predicted answer (simple extraction - take full content for now)
            predicted = response.content.strip()

            # Get ground truth
            ground_truth = sample.get("answer") or sample.get("final_answer", "")
            alternatives = sample.get("alternatives", [])

            # Calculate accuracy for this sample
            is_correct = calculate_accuracy(predicted, ground_truth, alternatives) > 0

            # Store prediction
            predictions.append({
                "sample_id": sample.get("id", f"sample_{i}"),
                "question": question,
                "prediction": predicted,
                "ground_truth": ground_truth,
                "correct": is_correct,
                "response_length": len(predicted),
                "tokens_used": response.tokens_used,
                "latency_ms": response.latency_ms,
            })

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"    • Processed {i+1}/{len(samples)} samples...")

        except Exception as e:
            print(f"    ⚠️  Error on sample {i}: {e}")
            predictions.append({
                "sample_id": sample.get("id", f"sample_{i}"),
                "question": question,
                "prediction": "",
                "ground_truth": sample.get("answer", ""),
                "correct": False,
                "error": str(e),
            })

    # Calculate metrics
    print(f"  → Calculating metrics for {technique_name}...")
    metrics = _calculate_metrics(predictions, responses, has_logprobs)

    return {
        "technique_name": technique_name,
        "predictions": predictions,
        "metrics": metrics,
        "sample_count": len(samples),
        "successful_count": sum(1 for p in predictions if "error" not in p),
        "has_logprobs": has_logprobs,
    }


def _calculate_metrics(
    predictions: List[Dict[str, Any]],
    responses: List[LLMResponse],
    has_logprobs: bool,
) -> Dict[str, Any]:
    """
    Calculate all metrics for a set of predictions.

    Args:
        predictions: List of prediction dictionaries
        responses: List of LLM responses
        has_logprobs: Whether responses have logprobs

    Returns:
        Dictionary with all computed metrics
    """
    # Calculate accuracy
    correct_count = sum(1 for p in predictions if p.get("correct", False))
    total_count = len(predictions)
    accuracy = correct_count / total_count if total_count > 0 else 0.0

    # Get response lengths
    response_lengths = [p.get("response_length", 0) for p in predictions]
    avg_response_length = int(np.mean(response_lengths)) if response_lengths else 0

    # Get prediction texts for diversity calculation
    pred_texts = [p.get("prediction", "") for p in predictions]

    metrics = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "avg_response_length": avg_response_length,
        "metrics_estimated": not has_logprobs,
    }

    if has_logprobs:
        # Use real logprobs for metrics
        all_logprobs = [r.logprobs for r in responses if r.logprobs]

        if all_logprobs:
            # Calculate average entropy
            entropies = []
            for logprobs in all_logprobs:
                entropy = calculate_entropy(logprobs=logprobs)
                entropies.append(entropy)

            avg_entropy = np.mean(entropies) if entropies else 0.0
            metrics["entropy"] = avg_entropy
            metrics["perplexity"] = calculate_perplexity(avg_entropy)

            # Calculate loss
            metrics["loss"] = calculate_loss(
                entropy=avg_entropy,
                response_length=avg_response_length,
                perplexity=metrics["perplexity"],
                accuracy=accuracy,
                normalize=True,
            )
        else:
            # Fallback even though logprobs were requested
            metrics.update(_calculate_fallback_metrics(
                pred_texts, accuracy, response_lengths
            ))
            metrics["metrics_estimated"] = True
    else:
        # Use fallback metrics
        metrics.update(_calculate_fallback_metrics(
            pred_texts, accuracy, response_lengths
        ))

    return metrics


def _calculate_fallback_metrics(
    predictions: List[str],
    accuracy: float,
    response_lengths: List[int],
) -> Dict[str, float]:
    """
    Calculate fallback metrics when logprobs are not available.

    Args:
        predictions: List of prediction texts
        accuracy: Overall accuracy
        response_lengths: List of response lengths

    Returns:
        Dictionary with fallback metrics
    """
    # Calculate fallback entropy
    entropy = calculate_fallback_entropy(
        predictions=predictions,
        accuracy=accuracy,
        response_lengths=response_lengths,
    )

    # Calculate fallback perplexity
    perplexity = calculate_fallback_perplexity(entropy)

    # Calculate average response length
    avg_length = int(np.mean(response_lengths)) if response_lengths else 0

    # Calculate fallback loss
    loss = calculate_fallback_loss(
        accuracy=accuracy,
        avg_response_length=avg_length,
        entropy=entropy,
        perplexity=perplexity,
    )

    return {
        "entropy": entropy,
        "perplexity": perplexity,
        "loss": loss,
    }


def collect_dataset_results(
    dataset_name: str,
    dataset: Dict[str, Any],
    technique_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Collect and organize results for a dataset.

    Args:
        dataset_name: Name of the dataset
        dataset: Dataset dictionary
        technique_results: List of technique evaluation results

    Returns:
        Organized results dictionary
    """
    # Find best and worst performing techniques
    sorted_results = sorted(
        technique_results,
        key=lambda x: x.get("metrics", {}).get("accuracy", 0.0),
        reverse=True
    )

    best_technique = sorted_results[0] if sorted_results else None
    worst_technique = sorted_results[-1] if len(sorted_results) > 1 else None

    return {
        "dataset_name": dataset_name,
        "total_samples": dataset.get("total_samples", 0),
        "categories": dataset.get("categories", []),
        "techniques_evaluated": len(technique_results),
        "best_technique": best_technique["technique_name"] if best_technique else None,
        "best_accuracy": best_technique.get("metrics", {}).get("accuracy", 0.0) if best_technique else 0.0,
        "worst_technique": worst_technique["technique_name"] if worst_technique else None,
        "worst_accuracy": worst_technique.get("metrics", {}).get("accuracy", 0.0) if worst_technique else 0.0,
    }
