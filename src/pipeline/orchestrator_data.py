"""
Data Loading Module for Experiment Orchestrator

Handles Phase 1: Dataset Loading
"""

from typing import Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_dataset


def load_datasets(
    dataset_paths: Dict[str, str],
    results: Dict[str, Any]
) -> None:
    """
    Phase 1: Load datasets.

    Args:
        dataset_paths: Dict mapping dataset names to file paths
        results: Results dictionary to update (modified in-place)
    """
    for dataset_name, dataset_path in dataset_paths.items():
        print(f"  Loading {dataset_name}...")
        dataset = load_dataset(dataset_path)
        results["datasets"][dataset_name] = {
            "path": dataset_path,
            "total_samples": dataset["total_samples"],
            "categories": dataset.get("categories", []),
        }
        results["metadata"]["total_samples"] += dataset["total_samples"]
        print(f"    ✓ Loaded {dataset['total_samples']} samples")
