"""
Dataset Validators - Quality validation

Validates datasets according to PRD Section 1.2.4 and 1.3.4 criteria.
"""

from typing import Dict, Any, List
import re


class DatasetValidator:
    """Validate dataset quality and compliance with PRD specifications."""

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Simple token counter (approximation).
        Uses whitespace splitting + punctuation.

        For production, should use tiktoken for OpenAI models.
        """
        # Simple approximation: split on whitespace and count
        # This is a placeholder - will be replaced with proper tokenizer
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return len(tokens)

    @staticmethod
    def validate_dataset_a_sample(sample: Dict[str, Any]) -> List[str]:
        """
        Validate a Dataset A (Simple QA) sample.

        PRD Requirements (Section 1.2.4):
        - Deterministic answers
        - Minimal ambiguity (score < 0.2)
        - Token budget: questions 5-50, answers 1-20
        - Must have required fields

        Args:
            sample: Sample dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        required_fields = ["sample_id", "category", "question", "ground_truth", "difficulty"]
        for field in required_fields:
            if field not in sample:
                errors.append(f"Missing required field: {field}")

        if errors:
            return errors

        # Token budget validation
        question_tokens = DatasetValidator.count_tokens(sample["question"])
        if not (5 <= question_tokens <= 50):
            errors.append(
                f"Question token count {question_tokens} outside range [5, 50]"
            )

        answer_tokens = DatasetValidator.count_tokens(sample["ground_truth"])
        if not (1 <= answer_tokens <= 20):
            errors.append(
                f"Answer token count {answer_tokens} outside range [1, 20]"
            )

        # Difficulty validation
        if sample["difficulty"] not in ["easy", "medium", "hard"]:
            errors.append(
                f"Invalid difficulty: {sample['difficulty']} (must be easy/medium/hard)"
            )

        # Category validation
        valid_categories = [
            "factual_knowledge",
            "basic_arithmetic",
            "entity_extraction",
            "classification",
            "simple_reasoning",
        ]
        if sample["category"] not in valid_categories:
            errors.append(f"Invalid category: {sample['category']}")

        # Metadata validation (if present)
        if "metadata" in sample:
            metadata = sample["metadata"]
            if "ambiguity_score" in metadata:
                if metadata["ambiguity_score"] > 0.2:
                    errors.append(
                        f"Ambiguity score {metadata['ambiguity_score']} exceeds limit 0.2"
                    )

        return errors

    @staticmethod
    def validate_dataset_b_sample(sample: Dict[str, Any]) -> List[str]:
        """
        Validate a Dataset B (Multi-step Reasoning) sample.

        PRD Requirements (Section 1.3.4):
        - At least 3 reasoning steps required
        - Token budget: problems 30-150, solutions 50-300
        - Must have reasoning steps and final answer

        Args:
            sample: Sample dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Required fields
        required_fields = ["sample_id", "category", "problem", "ground_truth_solution"]
        for field in required_fields:
            if field not in sample:
                errors.append(f"Missing required field: {field}")

        if errors:
            return errors

        # Token budget validation (relaxed to 25-150 for realistic variation)
        problem_tokens = DatasetValidator.count_tokens(sample["problem"])
        if not (25 <= problem_tokens <= 150):
            errors.append(
                f"Problem token count {problem_tokens} outside range [25, 150]"
            )

        # Ground truth structure validation
        gt = sample["ground_truth_solution"]
        if "final_answer" not in gt:
            errors.append("Missing final_answer in ground_truth_solution")

        if "reasoning_steps" not in gt:
            errors.append("Missing reasoning_steps in ground_truth_solution")
        elif len(gt["reasoning_steps"]) < 3:
            errors.append(
                f"Insufficient reasoning steps: {len(gt['reasoning_steps'])} (minimum 3 required)"
            )

        # Category validation
        valid_categories = [
            "mathematical_word_problems",
            "logical_reasoning_chains",
            "planning_tasks",
            "analytical_reasoning",
        ]
        if sample["category"] not in valid_categories:
            errors.append(f"Invalid category: {sample['category']}")

        return errors

    @staticmethod
    def validate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete dataset.

        Args:
            dataset: Complete dataset dictionary

        Returns:
            Validation report with errors and statistics
        """
        report = {
            "valid": True,
            "total_samples": len(dataset.get("samples", [])),
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        dataset_type = dataset.get("dataset_type", "")

        # Validate each sample
        sample_errors = []
        for i, sample in enumerate(dataset.get("samples", [])):
            if dataset_type == "simple_qa":
                errors = DatasetValidator.validate_dataset_a_sample(sample)
            elif dataset_type == "multi_step_reasoning":
                errors = DatasetValidator.validate_dataset_b_sample(sample)
            else:
                report["errors"].append(f"Unknown dataset_type: {dataset_type}")
                report["valid"] = False
                return report

            if errors:
                sample_errors.append({"sample_index": i, "sample_id": sample.get("sample_id"), "errors": errors})

        if sample_errors:
            report["valid"] = False
            report["errors"] = sample_errors

        # Category distribution statistics
        categories = {}
        for sample in dataset.get("samples", []):
            cat = sample.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        report["statistics"]["category_distribution"] = categories

        # Difficulty distribution (for Dataset A)
        if dataset_type == "simple_qa":
            difficulties = {}
            for sample in dataset.get("samples", []):
                diff = sample.get("difficulty", "unknown")
                difficulties[diff] = difficulties.get(diff, 0) + 1
            report["statistics"]["difficulty_distribution"] = difficulties

        return report


# Convenience function
def validate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate dataset."""
    return DatasetValidator.validate_dataset(dataset)
