"""
Unit tests for data module

Tests dataset creation, validation, and loading.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import (
    DatasetCreator,
    create_dataset_a,
    create_dataset_b,
    validate_dataset,
    DatasetValidator,
)


class TestDatasetCreator:
    """Test dataset creation functionality."""

    def test_create_dataset_a(self):
        """Test Dataset A creation."""
        dataset = create_dataset_a()

        assert dataset is not None
        assert dataset["dataset_id"] == "dataset_a_v1"
        assert dataset["dataset_type"] == "simple_qa"
        assert dataset["total_samples"] == 75
        assert len(dataset["samples"]) == 75

        # Check categories
        expected_categories = [
            "factual_knowledge",
            "basic_arithmetic",
            "entity_extraction",
            "classification",
            "simple_reasoning",
        ]
        assert dataset["categories"] == expected_categories

        # Check all samples have required fields
        for sample in dataset["samples"]:
            assert "sample_id" in sample
            assert "category" in sample
            assert "question" in sample
            assert "ground_truth" in sample
            assert "difficulty" in sample
            assert sample["category"] in expected_categories

    def test_create_dataset_b(self):
        """Test Dataset B creation."""
        dataset = create_dataset_b()

        assert dataset is not None
        assert dataset["dataset_id"] == "dataset_b_v1"
        assert dataset["dataset_type"] == "multi_step_reasoning"
        assert dataset["total_samples"] == 35
        assert len(dataset["samples"]) == 35

        # Check categories
        expected_categories = [
            "mathematical_word_problems",
            "logical_reasoning_chains",
            "planning_tasks",
            "analytical_reasoning",
        ]
        assert dataset["categories"] == expected_categories

        # Check all samples have required fields
        for sample in dataset["samples"]:
            assert "sample_id" in sample
            assert "category" in sample
            assert "problem" in sample
            assert "ground_truth_solution" in sample
            assert sample["category"] in expected_categories

            # Check ground truth structure
            gt = sample["ground_truth_solution"]
            assert "final_answer" in gt
            assert "reasoning_steps" in gt
            assert "step_count" in gt
            assert len(gt["reasoning_steps"]) >= 3  # Minimum 3 steps

    def test_category_distribution_dataset_a(self):
        """Test that Dataset A has correct category distribution."""
        dataset = create_dataset_a()
        categories = {}

        for sample in dataset["samples"]:
            cat = sample["category"]
            categories[cat] = categories.get(cat, 0) + 1

        # PRD requirements: 18, 18, 18, 12, 9
        assert categories["factual_knowledge"] == 18
        assert categories["basic_arithmetic"] == 18
        assert categories["entity_extraction"] == 18
        assert categories["classification"] == 12
        assert categories["simple_reasoning"] == 9

    def test_category_distribution_dataset_b(self):
        """Test that Dataset B has correct category distribution."""
        dataset = create_dataset_b()
        categories = {}

        for sample in dataset["samples"]:
            cat = sample["category"]
            categories[cat] = categories.get(cat, 0) + 1

        # PRD requirements: 11, 9, 9, 6
        assert categories["mathematical_word_problems"] == 11
        assert categories["logical_reasoning_chains"] == 9
        assert categories["planning_tasks"] == 9
        assert categories["analytical_reasoning"] == 6


class TestDatasetValidator:
    """Test dataset validation functionality."""

    def test_validate_dataset_a_sample_valid(self):
        """Test validation of a valid Dataset A sample."""
        sample = {
            "sample_id": "qa_001",
            "category": "factual_knowledge",
            "question": "What is the capital of France?",
            "ground_truth": "Paris",
            "difficulty": "easy",
            "metadata": {
                "tokens_question": 7,
                "tokens_answer": 1,
                "ambiguity_score": 0.0,
            },
        }

        errors = DatasetValidator.validate_dataset_a_sample(sample)
        assert len(errors) == 0

    def test_validate_dataset_a_sample_missing_field(self):
        """Test validation of a Dataset A sample with missing field."""
        sample = {
            "sample_id": "qa_001",
            "category": "factual_knowledge",
            "question": "What is the capital of France?",
            # Missing ground_truth
            "difficulty": "easy",
        }

        errors = DatasetValidator.validate_dataset_a_sample(sample)
        assert len(errors) > 0
        assert any("ground_truth" in err for err in errors)

    def test_validate_dataset_b_sample_valid(self):
        """Test validation of a valid Dataset B sample."""
        sample = {
            "sample_id": "msr_001",
            "category": "mathematical_word_problems",
            "problem": "A store offers a 20% discount on an item originally priced at $150. If sales tax is 8%, what is the final price?",
            "ground_truth_solution": {
                "final_answer": "$129.60",
                "reasoning_steps": ["Step 1", "Step 2", "Step 3", "Step 4"],
                "step_count": 4,
            },
        }

        errors = DatasetValidator.validate_dataset_b_sample(sample)
        assert len(errors) == 0

    def test_validate_dataset_b_sample_insufficient_steps(self):
        """Test validation of Dataset B sample with insufficient steps."""
        sample = {
            "sample_id": "msr_001",
            "category": "mathematical_word_problems",
            "problem": "A store offers a 20% discount...",
            "ground_truth_solution": {
                "final_answer": "$129.60",
                "reasoning_steps": ["Step 1", "Step 2"],  # Only 2 steps (< 3 minimum)
                "step_count": 2,
            },
        }

        errors = DatasetValidator.validate_dataset_b_sample(sample)
        assert len(errors) > 0
        assert any("reasoning steps" in err.lower() for err in errors)

    def test_validate_complete_dataset_a(self):
        """Test validation of complete Dataset A."""
        dataset = create_dataset_a()
        report = validate_dataset(dataset)

        assert report["valid"] is True
        assert report["total_samples"] == 75
        assert len(report["errors"]) == 0

    def test_validate_complete_dataset_b(self):
        """Test validation of complete Dataset B."""
        dataset = create_dataset_b()
        report = validate_dataset(dataset)

        assert report["valid"] is True
        assert report["total_samples"] == 35
        assert len(report["errors"]) == 0


# Run tests if executed directly
if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
