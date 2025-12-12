"""
Accuracy Metrics

Implements accuracy calculators for evaluating LLM responses.
"""

from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Import fuzzy_match utility
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm.utils import fuzzy_match


def calculate_exact_match(predicted: str, ground_truth: str) -> bool:
    """
    Check exact match (case-insensitive, whitespace-normalized).

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        True if exact match
    """
    return predicted.strip().lower() == ground_truth.strip().lower()


def calculate_accuracy(
    predicted: str,
    ground_truth: str,
    alternatives: Optional[List[str]] = None,
    use_fuzzy: bool = True,
) -> float:
    """
    Calculate accuracy for a single prediction.

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        alternatives: List of alternative acceptable answers
        use_fuzzy: Whether to use fuzzy matching

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    if use_fuzzy:
        return 1.0 if fuzzy_match(predicted, ground_truth, alternatives) else 0.0
    else:
        return 1.0 if calculate_exact_match(predicted, ground_truth) else 0.0


def calculate_dataset_accuracy(
    predictions: List[str],
    ground_truths: List[str],
    alternatives_list: Optional[List[List[str]]] = None,
    use_fuzzy: bool = True,
) -> Dict[str, Any]:
    """
    Calculate accuracy metrics for a dataset.

    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        alternatives_list: List of alternative answer lists (one per sample)
        use_fuzzy: Whether to use fuzzy matching

    Returns:
        Dictionary with accuracy metrics:
        - accuracy: Overall accuracy (0.0 to 1.0)
        - correct_count: Number of correct predictions
        - total_count: Total number of predictions
        - per_sample_accuracy: List of per-sample accuracy scores
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(
            f"Predictions and ground truths must have same length: "
            f"{len(predictions)} vs {len(ground_truths)}"
        )

    n = len(predictions)
    per_sample_accuracy = []
    correct_count = 0

    for i in range(n):
        predicted = predictions[i]
        ground_truth = ground_truths[i]

        # Get alternatives for this sample
        alternatives = None
        if alternatives_list and i < len(alternatives_list):
            alternatives = alternatives_list[i]

        # Calculate accuracy for this sample
        acc = calculate_accuracy(predicted, ground_truth, alternatives, use_fuzzy)
        per_sample_accuracy.append(acc)

        if acc > 0:
            correct_count += 1

    overall_accuracy = correct_count / n if n > 0 else 0.0

    return {
        "accuracy": overall_accuracy,
        "correct_count": correct_count,
        "total_count": n,
        "per_sample_accuracy": per_sample_accuracy,
    }


def calculate_multi_step_accuracy(
    predicted_steps: List[str],
    ground_truth_steps: List[str],
    partial_credit: bool = True,
) -> Dict[str, Any]:
    """
    Calculate accuracy for multi-step reasoning problems.

    For Dataset B (multi-step reasoning).

    Args:
        predicted_steps: List of predicted reasoning steps
        ground_truth_steps: List of ground truth reasoning steps
        partial_credit: Whether to give partial credit for partial correctness

    Returns:
        Dictionary with metrics:
        - final_accuracy: Accuracy of final answer
        - step_accuracy: Average accuracy across steps
        - correct_steps: Number of correct steps
        - total_steps: Total number of steps
    """
    n_predicted = len(predicted_steps)
    n_ground_truth = len(ground_truth_steps)

    if partial_credit:
        # Compare each step and give partial credit
        correct_steps = 0
        min_steps = min(n_predicted, n_ground_truth)

        for i in range(min_steps):
            if fuzzy_match(predicted_steps[i], ground_truth_steps[i]):
                correct_steps += 1

        # Final answer is last step
        final_accuracy = 1.0 if fuzzy_match(
            predicted_steps[-1] if predicted_steps else "",
            ground_truth_steps[-1] if ground_truth_steps else ""
        ) else 0.0

        step_accuracy = correct_steps / n_ground_truth if n_ground_truth > 0 else 0.0

        return {
            "final_accuracy": final_accuracy,
            "step_accuracy": step_accuracy,
            "correct_steps": correct_steps,
            "total_steps": n_ground_truth,
        }

    else:
        # All-or-nothing: all steps must be correct
        if n_predicted != n_ground_truth:
            return {
                "final_accuracy": 0.0,
                "step_accuracy": 0.0,
                "correct_steps": 0,
                "total_steps": n_ground_truth,
            }

        all_correct = all(
            fuzzy_match(predicted_steps[i], ground_truth_steps[i])
            for i in range(n_ground_truth)
        )

        return {
            "final_accuracy": 1.0 if all_correct else 0.0,
            "step_accuracy": 1.0 if all_correct else 0.0,
            "correct_steps": n_ground_truth if all_correct else 0,
            "total_steps": n_ground_truth,
        }
