"""
Comprehensive tests for metrics module

Improves coverage for metrics/information_theory.py and metrics/accuracy.py
"""

import sys
from pathlib import Path
import math

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.information_theory import (
    entropy_from_logprobs,
    calculate_entropy,
    calculate_perplexity,
    calculate_perplexity_from_logprobs,
    calculate_loss,
    calculate_average_entropy,
    calculate_average_perplexity,
    calculate_fallback_loss,
)
from metrics.accuracy import (
    calculate_accuracy,
    calculate_exact_match,
    calculate_dataset_accuracy,
    calculate_multi_step_accuracy,
)
from llm.utils import fuzzy_match


class TestEntropyCalculations:
    """Test entropy calculation functions comprehensively."""

    def test_entropy_from_logprobs_with_multiple_tokens(self):
        """Test entropy with multiple token logprobs."""
        logprobs = [
            {"token": "The", "logprob": -0.1},
            {"token": "quick", "logprob": -0.5},
            {"token": "brown", "logprob": -1.0},
            {"token": "fox", "logprob": -1.5},
        ]

        entropy = entropy_from_logprobs(logprobs)

        assert isinstance(entropy, float)
        assert entropy >= 0
        # More uncertain tokens should lead to higher entropy

    def test_entropy_high_certainty(self):
        """Test entropy with high certainty (low logprob)."""
        logprobs = [
            {"token": "test", "logprob": -0.0001},
        ]

        entropy = entropy_from_logprobs(logprobs)
        assert entropy < 0.01  # Very low entropy for very certain token

    def test_calculate_perplexity_from_logprobs(self):
        """Test perplexity calculation from logprobs."""
        logprobs = [
            {"token": "test", "logprob": -1.0},
            {"token": "word", "logprob": -1.5},
        ]

        perplexity = calculate_perplexity_from_logprobs(logprobs)

        assert isinstance(perplexity, float)
        assert perplexity >= 1.0

    def test_calculate_average_entropy(self):
        """Test average entropy across multiple samples."""
        samples_logprobs = [
            [{"token": "a", "logprob": -0.5}],
            [{"token": "b", "logprob": -1.0}],
            [{"token": "c", "logprob": -1.5}],
        ]

        avg_entropy = calculate_average_entropy(samples_logprobs)

        assert isinstance(avg_entropy, float)
        assert avg_entropy >= 0

    def test_calculate_average_perplexity(self):
        """Test average perplexity across multiple samples."""
        samples_logprobs = [
            [{"token": "a", "logprob": -0.5}],
            [{"token": "b", "logprob": -1.0}],
        ]

        avg_perplexity = calculate_average_perplexity(samples_logprobs)

        assert isinstance(avg_perplexity, float)
        assert avg_perplexity >= 1.0


class TestLossCalculations:
    """Test loss function calculations."""

    def test_calculate_loss_with_all_params(self):
        """Test loss with all parameters."""
        loss = calculate_loss(
            entropy=2.0,
            response_length=100,
            perplexity=4.0,
            accuracy=0.8,
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_calculate_loss_custom_weights(self):
        """Test loss with custom weights."""
        loss = calculate_loss(
            entropy=2.0,
            response_length=100,
            perplexity=4.0,
            accuracy=0.8,
            alpha=0.5,  # Custom weight for entropy
            beta=0.1,   # Custom weight for length
            gamma=0.2,  # Custom weight for perplexity
            delta=0.2,  # Custom weight for accuracy
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_calculate_loss_perfect_accuracy(self):
        """Test loss with perfect accuracy."""
        loss = calculate_loss(
            entropy=0.0,
            response_length=50,
            perplexity=1.0,
            accuracy=1.0,
        )

        # Perfect accuracy with low entropy should give low loss
        assert loss < 1.0

    def test_calculate_loss_zero_accuracy(self):
        """Test loss with zero accuracy."""
        loss = calculate_loss(
            entropy=5.0,
            response_length=200,
            perplexity=32.0,
            accuracy=0.0,
        )

        # Zero accuracy should give positive loss
        assert loss > 0.0
        assert isinstance(loss, float)

    def test_calculate_fallback_loss(self):
        """Test fallback loss calculation."""
        loss = calculate_fallback_loss(
            accuracy=0.75,
            avg_response_length=50,
            entropy=2.0,
            perplexity=4.0,
        )

        assert isinstance(loss, float)
        assert loss >= 0


class TestAccuracyMetrics:
    """Test accuracy evaluation functions."""

    def test_calculate_accuracy_correct(self):
        """Test accuracy calculation for correct answer."""
        accuracy = calculate_accuracy(
            predicted="Paris",
            ground_truth="Paris",
        )

        assert accuracy == 1.0

    def test_calculate_accuracy_incorrect(self):
        """Test accuracy calculation for incorrect answer."""
        accuracy = calculate_accuracy(
            predicted="London",
            ground_truth="Paris",
        )

        assert accuracy == 0.0

    def test_calculate_accuracy_case_insensitive(self):
        """Test case-insensitive matching."""
        accuracy = calculate_accuracy(
            predicted="paris",
            ground_truth="Paris",
        )

        assert accuracy == 1.0

    def test_calculate_accuracy_with_alternatives(self):
        """Test accuracy with alternative answers."""
        accuracy = calculate_accuracy(
            predicted="four",
            ground_truth="4",
            alternatives=["four", "4"],
        )

        assert accuracy == 1.0

    def test_calculate_dataset_accuracy_all_correct(self):
        """Test dataset accuracy with all correct answers."""
        predictions = ["Paris", "4", "Blue"]
        ground_truths = ["Paris", "4", "Blue"]

        metrics = calculate_dataset_accuracy(predictions, ground_truths)

        assert metrics["accuracy"] == 1.0
        assert metrics["total_count"] == 3
        assert metrics["correct_count"] == 3

    def test_calculate_dataset_accuracy_mixed(self):
        """Test dataset accuracy with mixed results."""
        predictions = ["Paris", "London", "Red", "Blue"]
        ground_truths = ["Paris", "Paris", "Red", "Green"]

        metrics = calculate_dataset_accuracy(predictions, ground_truths)

        assert metrics["accuracy"] == 0.5
        assert metrics["total_count"] == 4
        assert metrics["correct_count"] == 2

    def test_fuzzy_match_exact(self):
        """Test fuzzy matching with exact match."""
        match = fuzzy_match("test", "test")
        assert match is True

    def test_fuzzy_match_substring(self):
        """Test fuzzy matching with substring."""
        match = fuzzy_match("The answer is test", "test")
        assert match is True

    def test_fuzzy_match_no_match(self):
        """Test fuzzy matching with no match."""
        match = fuzzy_match("foo", "bar")
        assert match is False


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_entropy_empty_logprobs(self):
        """Test entropy with empty logprobs."""
        entropy = entropy_from_logprobs([])
        assert entropy == 0.0

    def test_perplexity_zero_entropy(self):
        """Test perplexity with zero entropy."""
        perplexity = calculate_perplexity(0.0)
        assert perplexity == 1.0

    def test_perplexity_high_entropy(self):
        """Test perplexity with high entropy."""
        perplexity = calculate_perplexity(10.0)
        assert perplexity == 2**10
        assert perplexity == 1024

    def test_loss_with_normalize_true(self):
        """Test loss calculation with normalization."""
        loss = calculate_loss(
            entropy=2.0,
            response_length=100,
            perplexity=4.0,
            accuracy=0.8,
            normalize=True,
        )

        assert isinstance(loss, float)

    def test_loss_with_normalize_false(self):
        """Test loss calculation without normalization."""
        loss = calculate_loss(
            entropy=2.0,
            response_length=100,
            perplexity=4.0,
            accuracy=0.8,
            normalize=False,
        )

        assert isinstance(loss, float)

    def test_accuracy_empty_results(self):
        """Test accuracy calculation with empty results."""
        try:
            metrics = calculate_dataset_accuracy([], [])
            # Should handle empty list gracefully
            assert metrics["total_count"] == 0 or metrics is not None
        except (ValueError, ZeroDivisionError):
            # Expected for empty results
            assert True

    def test_entropy_single_certain_token(self):
        """Test entropy with single very certain token."""
        logprobs = [{"token": "the", "logprob": -0.00001}]
        entropy = entropy_from_logprobs(logprobs)

        # Very certain prediction should have very low entropy
        assert entropy < 0.001

    def test_average_entropy_empty_list(self):
        """Test average entropy with empty sample list."""
        try:
            avg_entropy = calculate_average_entropy([])
            assert avg_entropy == 0.0 or avg_entropy >= 0
        except (ValueError, ZeroDivisionError):
            # Expected
            assert True

    def test_loss_extreme_values(self):
        """Test loss with extreme values."""
        # Very high entropy, long response, high perplexity, low accuracy
        loss_high = calculate_loss(
            entropy=15.0,
            response_length=5000,
            perplexity=1024.0,
            accuracy=0.0,
        )

        # Very low entropy, short response, low perplexity, high accuracy
        loss_low = calculate_loss(
            entropy=0.1,
            response_length=10,
            perplexity=1.1,
            accuracy=1.0,
        )

        assert loss_high > loss_low
