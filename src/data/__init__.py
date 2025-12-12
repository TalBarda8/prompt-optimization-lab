"""
Data Module - Dataset Creation and Validation

This module implements dataset creation, validation, and loading
as specified in PRD Section 1: Dataset Creation.

Components:
- dataset_creator: Generate Dataset A (Simple QA) and Dataset B (Multi-step Reasoning)
- validators: Quality validation (ambiguity scores, token counts, difficulty distribution)
- loaders: JSON loading/saving utilities with version control
"""

from .dataset_creator import DatasetCreator, create_dataset_a, create_dataset_b
from .validators import DatasetValidator, validate_dataset
from .loaders import DatasetLoader, load_dataset, save_dataset

__all__ = [
    "DatasetCreator",
    "create_dataset_a",
    "create_dataset_b",
    "DatasetValidator",
    "validate_dataset",
    "DatasetLoader",
    "load_dataset",
    "save_dataset",
]
