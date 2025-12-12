#!/usr/bin/env python3
"""
Quick Experiment Runner

Runs a simplified experiment with fewer samples for testing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import ExperimentConfig, ExperimentOrchestrator


def main():
    """Run quick experiment."""
    print("=" * 70)
    print("QUICK EXPERIMENT (Testing Mode)")
    print("=" * 70)
    print("Running a simplified experiment with subset of techniques...")
    print()

    config = ExperimentConfig(
        dataset_paths={
            "dataset_a": "data/dataset_a.json",
        },
        llm_provider="openai",
        llm_model="gpt-3.5-turbo",  # Faster/cheaper model
        temperature=0.0,
        max_tokens=200,  # Shorter responses
        techniques=["baseline", "chain_of_thought"],  # Only 2 techniques
        output_dir="results/quick_experiment",
        save_intermediate=True,
    )

    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_full_pipeline()

    print("\n" + "=" * 70)
    print("QUICK EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"Results: {config.output_dir}/experiment_results.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
