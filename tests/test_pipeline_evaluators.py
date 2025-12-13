"""
Unit tests for pipeline evaluator modules

Tests evaluator, experiment_evaluator functionality.
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.evaluator import BaselineEvaluator, PromptOptimizationEvaluator
from prompts.techniques import BaselinePrompt, ChainOfThoughtPrompt


class TestBaselineEvaluator:
    """Test BaselineEvaluator class."""

    def setup_method(self):
        """Setup test data."""
        self.test_data = [
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "category": "arithmetic",
            },
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "category": "geography",
            },
        ]

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        # Mock LLM client
        class MockLLMClient:
            pass

        client = MockLLMClient()
        evaluator = BaselineEvaluator(llm_client=client)

        assert evaluator is not None


class TestPromptOptimizationEvaluator:
    """Test PromptOptimizationEvaluator class."""

    def setup_method(self):
        """Setup test data."""
        self.datasets = {
            "dataset_a": [
                {
                    "question": "What is 2 + 2?",
                    "answer": "4",
                    "category": "arithmetic",
                },
            ],
        }

        self.techniques = {
            "baseline": BaselinePrompt(),
            "cot": ChainOfThoughtPrompt(),
        }

    def test_evaluator_initialization(self):
        """Test optimization evaluator initialization."""
        # Mock LLM client
        class MockLLMClient:
            pass

        client = MockLLMClient()
        evaluator = PromptOptimizationEvaluator(llm_client=client)

        assert evaluator is not None


