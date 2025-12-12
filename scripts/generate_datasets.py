#!/usr/bin/env python3
"""
Generate Dataset A and Dataset B

Creates both datasets and saves them to data/ directory.
Run this script to initialize the datasets for the experiment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import create_dataset_a, create_dataset_b, save_dataset, validate_dataset


def main():
    """Generate and save both datasets."""
    print("=" * 60)
    print("Generating Datasets for Prompt Optimization Experiment")
    print("=" * 60)

    # Generate Dataset A
    print("\n📊 Creating Dataset A (Simple QA - 75 samples)...")
    dataset_a = create_dataset_a()

    # Validate Dataset A
    print("✓ Dataset A created")
    print("🔍 Validating Dataset A...")
    validation_a = validate_dataset(dataset_a)

    if validation_a["valid"]:
        print(f"✅ Dataset A is valid!")
        print(f"   - Total samples: {validation_a['total_samples']}")
        print(f"   - Categories: {validation_a['statistics']['category_distribution']}")
        print(f"   - Difficulty: {validation_a['statistics']['difficulty_distribution']}")
    else:
        print(f"❌ Dataset A has errors:")
        for error in validation_a["errors"]:
            print(f"   - {error}")
        return

    # Save Dataset A
    dataset_a_path = "data/dataset_a.json"
    print(f"\n💾 Saving Dataset A to {dataset_a_path}...")
    save_dataset(dataset_a, dataset_a_path)
    print(f"✅ Dataset A saved successfully")

    # Generate Dataset B
    print("\n📊 Creating Dataset B (Multi-step Reasoning - 35 samples)...")
    dataset_b = create_dataset_b()

    # Validate Dataset B
    print("✓ Dataset B created")
    print("🔍 Validating Dataset B...")
    validation_b = validate_dataset(dataset_b)

    if validation_b["valid"]:
        print(f"✅ Dataset B is valid!")
        print(f"   - Total samples: {validation_b['total_samples']}")
        print(f"   - Categories: {validation_b['statistics']['category_distribution']}")
    else:
        print(f"❌ Dataset B has errors:")
        for error in validation_b["errors"]:
            print(f"   - {error}")
        return

    # Save Dataset B
    dataset_b_path = "data/dataset_b.json"
    print(f"\n💾 Saving Dataset B to {dataset_b_path}...")
    save_dataset(dataset_b, dataset_b_path)
    print(f"✅ Dataset B saved successfully")

    print("\n" + "=" * 60)
    print("✅ All datasets generated and validated successfully!")
    print("=" * 60)
    print(f"\nDataset A: {dataset_a_path} ({validation_a['total_samples']} samples)")
    print(f"Dataset B: {dataset_b_path} ({validation_b['total_samples']} samples)")
    print(f"Total: {validation_a['total_samples'] + validation_b['total_samples']} samples")
    print("\nYou can now proceed to Phase 2: Baseline Evaluation")


if __name__ == "__main__":
    main()
