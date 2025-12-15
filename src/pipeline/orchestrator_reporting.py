"""
Reporting and Visualization Module for Experiment Orchestrator

Handles Phase 6:
- Visualization Generation
- Results Saving
"""

from typing import Dict, Any
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from visualization import generate_visualization_report


def generate_visualizations(
    results: Dict[str, Any],
    techniques: list,
    output_path: Path
) -> None:
    """
    Phase 6: Generate all visualizations.

    Args:
        results: Results dictionary (modified in-place)
        techniques: List of technique names
        output_path: Output directory path
    """
    print("  Generating visualizations...")

    # Import the new visualization module
    from visualization.visualization import generate_all_visualizations

    # Generate the 4 key visualizations
    figures_dir = generate_all_visualizations(
        results,
        output_dir=str(output_path / "figures")
    )

    # Also generate the legacy 12-visualization report (if needed)
    # Prepare visualization data for backward compatibility
    viz_data = {
        "techniques": techniques,
        "accuracy": {},
        "loss": {},
        "total_samples": results["metadata"]["total_samples"],
    }

    try:
        report_path = generate_visualization_report(
            viz_data,
            output_dir=str(output_path),
            include_plots=True,
        )
    except Exception as e:
        print(f"    ⚠️  Legacy visualizations skipped: {e}")
        report_path = None

    results["visualizations"] = {
        "figures_dir": figures_dir,
        "report_path": report_path if report_path else str(output_path / "figures"),
    }

    print(f"\n    ✓ All visualizations saved to {output_path / 'figures'}")


def save_results(
    results: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save comprehensive results to JSON.

    Args:
        results: Results dictionary
        output_path: Output directory path
    """
    results_file = output_path / "experiment_results.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved: {results_file}")
