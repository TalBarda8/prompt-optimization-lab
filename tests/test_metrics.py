"""
Unit tests for metrics module

Tests information-theoretic metrics and accuracy calculators.
"""

import sys
from pathlib import Path
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics import (
    calculate_entropy,
    calculate_perplexity,
    calculate_loss,
    entropy_from_logprobs,
    calculate_accuracy,
    calculate_exact_match,
    calculate_dataset_accuracy,
    calculate_multi_step_accuracy,
)


class TestInformationTheory:
    """Test information-theoretic metrics."""

    def test_entropy_from_logprobs_uniform(self):
        """Test entropy calculation with uniform distribution."""
        # Two tokens with equal probability (p=0.5 each)
        # log(0.5) ≈ -0.693
        logprobs = [
            {"token": "A", "logprob": math.log(0.5)},
            {"token": "B", "logprob": math.log(0.5)},
        ]

        entropy = entropy_from_logprobs(logprobs)

        # Expected entropy for uniform distribution: -Σ p*log₂(p) = -2*(0.5*log₂(0.5)) = 1 bit
        assert math.isclose(entropy, 1.0, rel_tol=0.1)

    def test_entropy_from_logprobs_deterministic(self):
        """Test entropy with deterministic distribution."""
        # One token with probability ≈ 1.0
        logprobs = [
            {"token": "A", "logprob": math.log(0.99)},
        ]

        entropy = entropy_from_logprobs(logprobs)

        # Deterministic should have very low entropy
        assert entropy < 0.1

    def test_entropy_from_probabilities(self):
        """Test entropy calculation from probabilities."""
        # Uniform distribution: [0.5, 0.5]
        probabilities = [0.5, 0.5]
        entropy = calculate_entropy(probabilities=probabilities)

        # Expected: 1 bit
        assert math.isclose(entropy, 1.0, rel_tol=0.01)

    def test_perplexity_calculation(self):
        """Test perplexity calculation."""
        # Entropy of 1 bit
        entropy = 1.0
        perplexity = calculate_perplexity(entropy)

        # Perplexity = 2^1 = 2
        assert math.isclose(perplexity, 2.0, rel_tol=0.01)

    def test_perplexity_high_entropy(self):
        """Test perplexity with high entropy."""
        # Entropy of 3 bits
        entropy = 3.0
        perplexity = calculate_perplexity(entropy)

        # Perplexity = 2^3 = 8
        assert math.isclose(perplexity, 8.0, rel_tol=0.01)

    def test_loss_function_basic(self):
        """Test loss function calculation."""
        # Basic test with normalized values
        loss = calculate_loss(
            entropy=5.0,  # bits
            response_length=100,  # tokens
            perplexity=10.0,
            accuracy=0.8,
            alpha=0.3,
            beta=0.2,
            gamma=0.2,
            delta=0.3,
            normalize=True,
        )

        # Loss should be between 0 and 1 when normalized
        assert 0.0 <= loss <= 1.0

    def test_loss_function_perfect(self):
        """Test loss function with perfect metrics."""
        loss = calculate_loss(
            entropy=0.0,  # No uncertainty
            response_length=0,  # Minimal length
            perplexity=1.0,  # Maximum confidence
            accuracy=1.0,  # Perfect accuracy
            normalize=True,
        )

        # Perfect metrics should give very low loss
        assert loss < 0.1

    def test_loss_function_weights_validation(self):
        """Test that loss function validates weight sum."""
        try:
            calculate_loss(
                entropy=1.0,
                response_length=10,
                perplexity=2.0,
                accuracy=0.5,
                alpha=0.5,
                beta=0.5,
                gamma=0.5,
                delta=0.5,  # Sum = 2.0, should fail
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "must sum to 1.0" in str(e)


class TestAccuracyMetrics:
    """Test accuracy calculation functions."""

    def test_exact_match_correct(self):
        """Test exact match with correct prediction."""
        assert calculate_exact_match("Paris", "Paris") is True
        assert calculate_exact_match("paris", "Paris") is True
        assert calculate_exact_match("  Paris  ", "Paris") is True

    def test_exact_match_incorrect(self):
        """Test exact match with incorrect prediction."""
        assert calculate_exact_match("London", "Paris") is False
        assert calculate_exact_match("Paris, France", "Paris") is False

    def test_accuracy_fuzzy_match(self):
        """Test accuracy with fuzzy matching."""
        # Exact match
        acc = calculate_accuracy("Paris", "Paris", use_fuzzy=True)
        assert acc == 1.0

        # Contained match
        acc = calculate_accuracy("The capital is Paris", "Paris", use_fuzzy=True)
        assert acc == 1.0

        # No match
        acc = calculate_accuracy("London", "Paris", use_fuzzy=True)
        assert acc == 0.0

    def test_accuracy_with_alternatives(self):
        """Test accuracy with alternative answers."""
        acc = calculate_accuracy(
            "USA",
            "United States",
            alternatives=["US", "USA", "America"],
            use_fuzzy=True,
        )
        assert acc == 1.0

    def test_dataset_accuracy(self):
        """Test dataset-level accuracy calculation."""
        predictions = ["Paris", "42", "London"]
        ground_truths = ["Paris", "42", "Madrid"]

        result = calculate_dataset_accuracy(predictions, ground_truths)

        assert result["accuracy"] == 2 / 3  # 2 out of 3 correct
        assert result["correct_count"] == 2
        assert result["total_count"] == 3
        assert len(result["per_sample_accuracy"]) == 3

    def test_dataset_accuracy_with_alternatives(self):
        """Test dataset accuracy with alternatives."""
        predictions = ["USA", "UK"]
        ground_truths = ["United States", "United Kingdom"]
        alternatives_list = [
            ["US", "USA", "America"],
            ["UK", "Britain"],
        ]

        result = calculate_dataset_accuracy(
            predictions, ground_truths, alternatives_list
        )

        assert result["accuracy"] == 1.0  # Both should match with alternatives
        assert result["correct_count"] == 2

    def test_multi_step_accuracy_full_credit(self):
        """Test multi-step accuracy with all steps correct."""
        predicted = ["Step 1: X", "Step 2: Y", "Step 3: Z"]
        ground_truth = ["Step 1: X", "Step 2: Y", "Step 3: Z"]

        result = calculate_multi_step_accuracy(predicted, ground_truth)

        assert result["final_accuracy"] == 1.0
        assert result["step_accuracy"] == 1.0
        assert result["correct_steps"] == 3

    def test_multi_step_accuracy_partial_credit(self):
        """Test multi-step accuracy with partial correctness."""
        predicted = ["Step 1: X", "Step 2: WRONG", "Step 3: Z"]
        ground_truth = ["Step 1: X", "Step 2: Y", "Step 3: Z"]

        result = calculate_multi_step_accuracy(
            predicted, ground_truth, partial_credit=True
        )

        assert result["step_accuracy"] == 2 / 3  # 2 out of 3 correct
        assert result["correct_steps"] == 2

    def test_multi_step_accuracy_final_wrong(self):
        """Test multi-step accuracy with wrong final answer."""
        predicted = ["Step 1: X", "Step 2: Y", "Step 3: WRONG"]
        ground_truth = ["Step 1: X", "Step 2: Y", "Step 3: Z"]

        result = calculate_multi_step_accuracy(predicted, ground_truth)

        assert result["final_accuracy"] == 0.0
        assert result["step_accuracy"] == 2 / 3


# Run tests if executed directly
if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
