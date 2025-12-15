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
        Validate a Dataset A (Simple QA / Complex QA) sample.

        PRD Requirements (Section 1.2.4):
        - Deterministic answers
        - Minimal ambiguity (score < 0.2)
        - Token budget: questions 5-50, answers 1-20 (relaxed for complex_qa)
        - Must have required fields

        Supports both old schema (sample_id, ground_truth, difficulty)
        and new schema (id, answer, category).

        Args:
            sample: Sample dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Support both old and new field names
        has_sample_id = "sample_id" in sample
        has_id = "id" in sample
        has_ground_truth = "ground_truth" in sample
        has_answer = "answer" in sample

        # Required fields (flexible schema)
        if not has_sample_id and not has_id:
            errors.append("Missing required field: sample_id or id")
        if "question" not in sample:
            errors.append("Missing required field: question")
        if not has_ground_truth and not has_answer:
            errors.append("Missing required field: ground_truth or answer")
        if "category" not in sample:
            errors.append("Missing required field: category")

        if errors:
            return errors

        # Now determine which field names to use
        sample_id_field = "sample_id" if has_sample_id else "id"
        answer_field = "ground_truth" if has_ground_truth else "answer"

        if errors:
            return errors

        # Token budget validation (relaxed for complex_qa: 5-150 for questions, 1-100 for answers)
        question_tokens = DatasetValidator.count_tokens(sample["question"])
        if not (5 <= question_tokens <= 150):
            errors.append(
                f"Question token count {question_tokens} outside range [5, 150]"
            )

        answer_tokens = DatasetValidator.count_tokens(sample[answer_field])
        if not (1 <= answer_tokens <= 100):
            errors.append(
                f"Answer token count {answer_tokens} outside range [1, 100]"
            )

        # Difficulty validation (optional field)
        if "difficulty" in sample:
            if sample["difficulty"] not in ["easy", "medium", "hard"]:
                errors.append(
                    f"Invalid difficulty: {sample['difficulty']} (must be easy/medium/hard)"
                )

        # Category validation (flexible - accept any category for complex_qa)
        # Old categories
        old_categories = [
            "factual_knowledge",
            "basic_arithmetic",
            "entity_extraction",
            "classification",
            "simple_reasoning",
        ]
        # New categories (for complex_qa)
        new_categories = [
            "ambiguous_queries",
            "adversarial_questions",
            "noisy_input",
            "multi_step_reasoning",
            "long_context",
            "factual_knowledge",
        ]
        all_valid_categories = old_categories + new_categories

        if sample["category"] not in all_valid_categories:
            # Don't error, just warn if category is not recognized
            pass  # Flexible validation

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
        Validate a Dataset B (Multi-step Reasoning / Complex Multi-step Reasoning) sample.

        PRD Requirements (Section 1.3.4):
        - At least 3 reasoning steps required (relaxed for complex_multi_step_reasoning)
        - Token budget: problems 30-150, solutions 50-300 (relaxed for complex datasets)
        - Must have reasoning steps and final answer (or simplified answer field)

        Supports both old schema (sample_id, problem, ground_truth_solution)
        and new schema (id, question, answer).

        Args:
            sample: Sample dictionary

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Support both old and new field names
        has_sample_id = "sample_id" in sample
        has_id = "id" in sample
        has_problem = "problem" in sample
        has_question = "question" in sample
        has_ground_truth_solution = "ground_truth_solution" in sample
        has_answer = "answer" in sample

        # Required fields (flexible schema)
        if not has_sample_id and not has_id:
            errors.append("Missing required field: sample_id or id")
        if not has_problem and not has_question:
            errors.append("Missing required field: problem or question")
        if not has_ground_truth_solution and not has_answer:
            errors.append("Missing required field: ground_truth_solution or answer")
        if "category" not in sample:
            errors.append("Missing required field: category")

        if errors:
            return errors

        # Now determine which field names to use
        sample_id_field = "sample_id" if has_sample_id else "id"
        question_field = "problem" if has_problem else "question"
        answer_field = "ground_truth_solution" if has_ground_truth_solution else "answer"

        # Token budget validation (relaxed: 5-200 for questions, 1-300 for answers)
        question_tokens = DatasetValidator.count_tokens(sample[question_field])
        if not (5 <= question_tokens <= 200):
            errors.append(
                f"Question token count {question_tokens} outside range [5, 200]"
            )

        # Ground truth structure validation (only for old schema with structured solution)
        if answer_field == "ground_truth_solution" and isinstance(sample[answer_field], dict):
            gt = sample[answer_field]
            if "final_answer" not in gt:
                errors.append("Missing final_answer in ground_truth_solution")

            if "reasoning_steps" not in gt:
                errors.append("Missing reasoning_steps in ground_truth_solution")
            elif len(gt["reasoning_steps"]) < 3:
                errors.append(
                    f"Insufficient reasoning steps: {len(gt['reasoning_steps'])} (minimum 3 required)"
                )
        # For new schema with simple answer field, just validate token count
        elif answer_field == "answer":
            answer_tokens = DatasetValidator.count_tokens(str(sample[answer_field]))
            if not (1 <= answer_tokens <= 300):
                errors.append(
                    f"Answer token count {answer_tokens} outside range [1, 300]"
                )

        # Category validation (flexible - accept both old and new categories)
        old_categories = [
            "mathematical_word_problems",
            "logical_reasoning_chains",
            "planning_tasks",
            "analytical_reasoning",
        ]
        new_categories = [
            "multi_step_reasoning",
            "planning_tasks",
            "adversarial_questions",
            "long_context",
            "ambiguous_queries",
            "noisy_input",
        ]
        all_valid_categories = old_categories + new_categories

        if sample["category"] not in all_valid_categories:
            # Don't error, just be flexible
            pass  # Flexible validation

        return errors

    @staticmethod
    def validate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate complete dataset.

        Supports both old dataset types (simple_qa, multi_step_reasoning)
        and new dataset types (complex_qa, complex_multi_step_reasoning).

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

        # Map new dataset types to validation functions
        # "complex_qa" -> use dataset_a validator
        # "complex_multi_step_reasoning" -> use dataset_b validator
        qa_types = ["simple_qa", "complex_qa"]
        reasoning_types = ["multi_step_reasoning", "complex_multi_step_reasoning"]

        # Validate each sample
        sample_errors = []
        for i, sample in enumerate(dataset.get("samples", [])):
            if dataset_type in qa_types:
                errors = DatasetValidator.validate_dataset_a_sample(sample)
            elif dataset_type in reasoning_types:
                errors = DatasetValidator.validate_dataset_b_sample(sample)
            else:
                report["errors"].append(f"Unknown dataset_type: {dataset_type}")
                report["valid"] = False
                return report

            if errors:
                # Support both old (sample_id) and new (id) field names
                sample_id = sample.get("sample_id") or sample.get("id") or f"sample_{i}"
                sample_errors.append({"sample_index": i, "sample_id": sample_id, "errors": errors})

        if sample_errors:
            report["valid"] = False
            report["errors"] = sample_errors

        # Category distribution statistics
        categories = {}
        for sample in dataset.get("samples", []):
            cat = sample.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1

        report["statistics"]["category_distribution"] = categories

        # Difficulty distribution (for QA datasets that have difficulty field)
        if dataset_type in qa_types:
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
