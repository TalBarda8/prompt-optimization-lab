"""
Simple tests to push coverage from 69% to 70%+

Targeting easy-to-test utility functions and edge cases.
"""

import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loaders import load_dataset, save_dataset
from metrics.accuracy import calculate_exact_match, calculate_multi_step_accuracy
from prompts.base import PromptTemplate, PromptTechnique


class TestDataLoadersSaveDataset:
    """Test save_dataset function."""

    def test_save_dataset_basic(self):
        """Test saving a dataset to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.json"
            dataset = {"samples": [{"q": "test", "a": "answer"}]}

            save_dataset(dataset, str(output_path))

            # Verify file was created
            assert output_path.exists()

            # Verify content is correct
            loaded = load_dataset(str(output_path))
            assert "samples" in loaded


class TestExactMatch:
    """Test calculate_exact_match function."""

    def test_exact_match_case_insensitive(self):
        """Test exact match is case insensitive."""
        assert calculate_exact_match("Paris", "paris") is True
        assert calculate_exact_match("PARIS", "paris") is True

    def test_exact_match_with_whitespace(self):
        """Test exact match handles whitespace."""
        assert calculate_exact_match("  Paris  ", "Paris") is True
        assert calculate_exact_match("Paris", "  Paris  ") is True

    def test_exact_match_different_strings(self):
        """Test exact match returns False for different strings."""
        assert calculate_exact_match("London", "Paris") is False


class TestMultiStepAccuracy:
    """Test calculate_multi_step_accuracy function."""

    def test_multi_step_all_correct_with_partial_credit(self):
        """Test multi-step accuracy when all steps correct."""
        predicted = ["step1", "step2", "step3"]
        ground_truth = ["step1", "step2", "step3"]

        metrics = calculate_multi_step_accuracy(
            predicted_steps=predicted,
            ground_truth_steps=ground_truth,
            partial_credit=True
        )

        assert metrics["final_accuracy"] == 1.0
        assert metrics["step_accuracy"] == 1.0

    def test_multi_step_partial_correct(self):
        """Test multi-step accuracy with some correct steps."""
        predicted = ["step1", "wrong", "step3"]
        ground_truth = ["step1", "step2", "step3"]

        metrics = calculate_multi_step_accuracy(
            predicted_steps=predicted,
            ground_truth_steps=ground_truth,
            partial_credit=True
        )

        assert metrics["final_accuracy"] == 1.0  # Final step correct
        assert metrics["correct_steps"] == 2
        assert metrics["total_steps"] == 3

    def test_multi_step_no_partial_credit_mismatch_length(self):
        """Test multi-step without partial credit and different lengths."""
        predicted = ["step1", "step2"]
        ground_truth = ["step1", "step2", "step3"]

        metrics = calculate_multi_step_accuracy(
            predicted_steps=predicted,
            ground_truth_steps=ground_truth,
            partial_credit=False
        )

        # Different lengths means all incorrect
        assert metrics["final_accuracy"] == 0.0
        assert metrics["step_accuracy"] == 0.0

    def test_multi_step_no_partial_credit_all_correct(self):
        """Test multi-step without partial credit when all correct."""
        predicted = ["step1", "step2"]
        ground_truth = ["step1", "step2"]

        metrics = calculate_multi_step_accuracy(
            predicted_steps=predicted,
            ground_truth_steps=ground_truth,
            partial_credit=False
        )

        assert metrics["final_accuracy"] == 1.0
        assert metrics["step_accuracy"] == 1.0


class TestPromptTemplate:
    """Test PromptTemplate class."""

    def test_prompt_template_creation(self):
        """Test creating a PromptTemplate."""
        template = PromptTemplate(
            technique=PromptTechnique.BASELINE,
            system_prompt="System",
            user_prompt="User: {question}",
            examples=None,
            metadata={"test": "value"}
        )

        assert template.technique == PromptTechnique.BASELINE
        assert template.system_prompt == "System"
        assert "question" in template.user_prompt
        assert template.metadata["test"] == "value"

    def test_prompt_template_without_system(self):
        """Test PromptTemplate without system prompt."""
        template = PromptTemplate(
            technique=PromptTechnique.CHAIN_OF_THOUGHT,
            system_prompt=None,
            user_prompt="Question: {question}",
            examples=None,
            metadata={}
        )

        assert template.system_prompt is None
        assert template.user_prompt is not None

    def test_prompt_template_with_examples(self):
        """Test PromptTemplate with examples."""
        examples = ["Example 1", "Example 2"]
        template = PromptTemplate(
            technique=PromptTechnique.FEW_SHOT,
            system_prompt=None,
            user_prompt="{question}",
            examples=examples,
            metadata={}
        )

        assert template.examples == examples
        assert len(template.examples) == 2


class TestPromptTechnique:
    """Test PromptTechnique enum."""

    def test_prompt_technique_values(self):
        """Test PromptTechnique enum has expected values."""
        assert PromptTechnique.BASELINE.value == "baseline"
        assert PromptTechnique.CHAIN_OF_THOUGHT.value == "chain_of_thought"
        assert PromptTechnique.REACT.value == "react"

    def test_prompt_technique_from_string(self):
        """Test creating PromptTechnique from string."""
        technique = PromptTechnique("baseline")
        assert technique == PromptTechnique.BASELINE

        technique2 = PromptTechnique("chain_of_thought")
        assert technique2 == PromptTechnique.CHAIN_OF_THOUGHT
