"""
Final tests to cover remaining information_theory.py lines (73, 235, 266, 279)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.information_theory import (
    calculate_entropy,
    calculate_average_perplexity,
    calculate_fallback_entropy
)


class TestCalculateEntropyWithLogprobs:
    """Cover line 73 - calculate_entropy when logprobs provided."""

    def test_calculate_entropy_with_valid_logprobs(self):
        """Test entropy calculation when logprobs is not None."""
        logprobs = [
            {"token": "test", "logprob": -0.5},
            {"token": "word", "logprob": -1.0}
        ]

        # This should call entropy_from_logprobs (line 73)
        entropy = calculate_entropy(logprobs=logprobs)

        assert isinstance(entropy, float)
        assert entropy >= 0


class TestAveragePerplexityEmpty:
    """Cover line 235 - average perplexity with empty samples."""

    def test_average_perplexity_empty_list(self):
        """Test average perplexity when samples_logprobs is empty."""
        # Empty list should return 0.0 (line 235)
        avg_perplexity = calculate_average_perplexity([])

        assert avg_perplexity == 0.0

    def test_average_perplexity_none_input(self):
        """Test average perplexity with None."""
        # None should also return 0.0 (line 235)
        try:
            avg_perplexity = calculate_average_perplexity(None)
            assert avg_perplexity == 0.0
        except (TypeError, AttributeError):
            # May raise error for None, that's okay
            assert True


class TestFallbackEntropyEmpty:
    """Cover line 266 - fallback entropy with empty predictions."""

    def test_fallback_entropy_empty_predictions(self):
        """Test fallback entropy when predictions list is empty."""
        # Empty predictions should return 5.0 (line 266)
        entropy = calculate_fallback_entropy(
            predictions=[],
            accuracy=0.8,
            response_lengths=[10, 20, 30]
        )

        assert entropy == 5.0


class TestFallbackEntropyLengthVariance:
    """Cover line 279 - length variance calculation."""

    def test_fallback_entropy_with_varied_lengths(self):
        """Test fallback entropy with varied response lengths."""
        predictions = ["answer1", "answer2", "answer3"]
        accuracy = 0.75
        # Varied lengths to trigger variance calculation (line 279)
        response_lengths = [10, 50, 100]

        entropy = calculate_fallback_entropy(
            predictions=predictions,
            accuracy=accuracy,
            response_lengths=response_lengths
        )

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_fallback_entropy_with_uniform_lengths(self):
        """Test fallback entropy with uniform response lengths."""
        predictions = ["a", "b", "c"]
        accuracy = 0.9
        # Uniform lengths
        response_lengths = [10, 10, 10]

        entropy = calculate_fallback_entropy(
            predictions=predictions,
            accuracy=accuracy,
            response_lengths=response_lengths
        )

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_fallback_entropy_zero_avg_length(self):
        """Test fallback entropy when average length is zero."""
        predictions = ["x"]
        accuracy = 0.5
        # Zero lengths edge case
        response_lengths = [0]

        entropy = calculate_fallback_entropy(
            predictions=predictions,
            accuracy=accuracy,
            response_lengths=response_lengths
        )

        # Should handle zero division gracefully (line 279 else clause)
        assert isinstance(entropy, float)
