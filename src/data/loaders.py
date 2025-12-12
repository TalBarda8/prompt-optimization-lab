"""
Dataset Loaders - JSON loading/saving utilities

Handles dataset persistence with version control and validation.
PRD Reference: Section 1.4 (Dataset Metadata & Versioning)
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class DatasetLoader:
    """Load and save datasets with version control."""

    @staticmethod
    def load_dataset(file_path: str) -> Dict[str, Any]:
        """
        Load a dataset from JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dataset dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        return dataset

    @staticmethod
    def save_dataset(
        dataset: Dict[str, Any],
        file_path: str,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> None:
        """
        Save a dataset to JSON file.

        Args:
            dataset: Dataset dictionary
            file_path: Path to save JSON file
            indent: JSON indentation (default: 2)
            ensure_ascii: Ensure ASCII encoding (default: False for Unicode support)
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata if not present
        if "metadata" not in dataset:
            dataset["metadata"] = {}

        dataset["metadata"]["last_modified"] = datetime.now().isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=indent, ensure_ascii=ensure_ascii)

    @staticmethod
    def validate_structure(dataset: Dict[str, Any], dataset_type: str) -> bool:
        """
        Validate dataset structure.

        Args:
            dataset: Dataset dictionary
            dataset_type: "simple_qa" or "multi_step_reasoning"

        Returns:
            True if valid structure
        """
        required_fields = ["dataset_id", "dataset_type", "total_samples", "categories", "samples"]

        for field in required_fields:
            if field not in dataset:
                raise ValueError(f"Missing required field: {field}")

        if dataset["dataset_type"] != dataset_type:
            raise ValueError(f"Expected dataset_type={dataset_type}, got {dataset['dataset_type']}")

        if len(dataset["samples"]) != dataset["total_samples"]:
            raise ValueError(
                f"Sample count mismatch: declared {dataset['total_samples']}, "
                f"found {len(dataset['samples'])}"
            )

        return True


# Convenience functions
def load_dataset(file_path: str) -> Dict[str, Any]:
    """Convenience function to load dataset."""
    return DatasetLoader.load_dataset(file_path)


def save_dataset(dataset: Dict[str, Any], file_path: str) -> None:
    """Convenience function to save dataset."""
    DatasetLoader.save_dataset(dataset, file_path)
