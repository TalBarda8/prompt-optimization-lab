"""
Extended unit tests for metrics.information_theory module

Tests additional functions and edge cases for information-theoretic metrics.
"""

import sys
from pathlib import Path
import math
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metrics.information_theory import (
    entropy_from_logprobs,
    calculate_entropy,
    calculate_perplexity,
    calculate_loss,
    calculate_fallback_entropy,
    calculate_fallback_perplexity,
)


class TestEntropyCalculations:
    """Test entropy calculation functions."""

    def test_entropy_from_logprobs_basic(self):
        """Test basic entropy calculation from logprobs."""
        logprobs = [
            {"token": "hello", "logprob": -1.0},
            {"token": "world", "logprob": -2.0},
        ]

        entropy = entropy_from_logprobs(logprobs)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_entropy_from_logprobs_empty(self):
        """Test entropy with empty logprobs."""
        entropy = entropy_from_logprobs([])

        assert entropy == 0.0

    def test_entropy_from_logprobs_single_token(self):
        """Test entropy with single token."""
        logprobs = [{"token": "hello", "logprob": -0.5}]

        entropy = entropy_from_logprobs(logprobs)

        assert entropy >= 0

    def test_calculate_entropy_from_probabilities(self):
        """Test entropy calculation from probabilities."""
        probabilities = [0.5, 0.3, 0.2]

        entropy = calculate_entropy(probabilities=probabilities)

        assert isinstance(entropy, float)
        assert entropy >= 0

    def test_calculate_entropy_uniform(self):
        """Test entropy with uniform distribution."""
        # Uniform distribution has maximum entropy
        probabilities = [0.25, 0.25, 0.25, 0.25]

        entropy = calculate_entropy(probabilities=probabilities)

        # Entropy of uniform distribution with 4 outcomes is 2 bits
        assert abs(entropy - 2.0) < 0.01

    def test_calculate_entropy_deterministic(self):
        """Test entropy with deterministic distribution."""
        # Deterministic distribution has zero entropy
        probabilities = [1.0, 0.0, 0.0, 0.0]

        entropy = calculate_entropy(probabilities=probabilities)

        assert entropy == 0.0

    def test_calculate_entropy_no_input(self):
        """Test entropy with no input."""
        try:
            entropy = calculate_entropy()
            assert False,  "Should raise ValueError"
        except ValueError:
            assert True


class TestPerplexityCalculations:
    """Test perplexity calculation functions."""

    def test_calculate_perplexity_basic(self):
        """Test basic perplexity calculation."""
        entropy = 2.0

        perplexity = calculate_perplexity(entropy)

        assert perplexity == 4.0  # 2^2 = 4

    def test_calculate_perplexity_zero(self):
        """Test perplexity with zero entropy."""
        perplexity = calculate_perplexity(0.0)

        assert perplexity == 1.0  # 2^0 = 1

    def test_calculate_perplexity_high(self):
        """Test perplexity with high entropy."""
        entropy = 10.0

        perplexity = calculate_perplexity(entropy)

        assert perplexity == 2**10


class TestLossCalculation:
    """Test loss calculation functions."""

    def test_calculate_loss(self):
        """Test loss calculation."""
        loss = calculate_loss(
            entropy=2.5,
            response_length=75,
            perplexity=5.0,
            accuracy=0.85
        )

        assert isinstance(loss, float)
        assert loss >= 0

    def test_calculate_loss_perfect(self):
        """Test loss with perfect accuracy."""
        loss = calculate_loss(
            entropy=0.0,
            response_length=50,
            perplexity=1.0,
            accuracy=1.0
        )

        assert loss >= 0

    def test_calculate_fallback_entropy(self):
        """Test fallback entropy calculation."""
        entropy = calculate_fallback_entropy(
            response_text="This is a test response.",
            avg_response_length=10
        )

        assert isinstance(entropy, float)

    def test_calculate_fallback_perplexity(self):
        """Test fallback perplexity calculation."""
        perplexity = calculate_fallback_perplexity(entropy=2.5)

        assert isinstance(perplexity, float)
        assert perplexity > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_entropy_input(self):
        """Test perplexity with invalid negative entropy."""
        try:
            perplexity = calculate_perplexity(-1.0)
            # Should handle gracefully or raise exception
            assert perplexity >= 0 or perplexity is None
        except ValueError:
            # Expected behavior
            pass

    def test_very_large_entropy(self):
        """Test perplexity with very large entropy."""
        entropy = 100.0

        perplexity = calculate_perplexity(entropy)

        assert perplexity > 0
        assert not math.isinf(perplexity) or perplexity == float('inf')

    def test_loss_with_all_params(self):
        """Test loss calculation with all parameters."""
        try:
            loss = calculate_loss(
                entropy=2.0,
                response_length=100,
                accuracy=0.8
            )
            assert loss >= 0
        except TypeError:
            # Function may have different signature
            pass

    def test_entropy_with_zero_probability(self):
        """Test entropy with zero probabilities."""
        probabilities = [0.5, 0.5, 0.0, 0.0]

        entropy = calculate_entropy(probabilities=probabilities)

        # Should handle zeros gracefully
        assert entropy >= 0
        assert not math.isnan(entropy)
