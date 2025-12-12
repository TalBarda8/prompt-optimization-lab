"""
Evaluation Metrics Module

Implements information-theoretic metrics and accuracy calculators
according to PRD Section 2.3.
"""

from .information_theory import (
    calculate_entropy,
    calculate_perplexity,
    calculate_loss,
    entropy_from_logprobs,
)
from .accuracy import (
    calculate_accuracy,
    calculate_exact_match,
    calculate_dataset_accuracy,
    calculate_multi_step_accuracy,
)

__all__ = [
    # Information theory metrics
    "calculate_entropy",
    "calculate_perplexity",
    "calculate_loss",
    "entropy_from_logprobs",
    # Accuracy metrics
    "calculate_accuracy",
    "calculate_exact_match",
    "calculate_dataset_accuracy",
    "calculate_multi_step_accuracy",
]
