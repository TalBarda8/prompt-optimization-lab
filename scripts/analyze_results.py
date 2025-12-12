#!/usr/bin/env python3
"""
Results Analyzer

Analyzes and summarizes experimental results.
"""

import sys
from pathlib import Path
import json
import argparse


def analyze_results(results_path: str):
    """Analyze experimental results."""
    print("=" * 70)
    print("EXPERIMENT RESULTS ANALYSIS")
    print("=" * 70)

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Print configuration
    print("\n📋 Configuration:")
    print(f"  Model: {results['config']['llm_model']}")
    print(f"  Provider: {results['config']['llm_provider']}")
    print(f"  Techniques: {', '.join(results['config']['techniques'])}")

    # Print dataset info
    print("\n📊 Datasets:")
    for dataset_name, dataset_info in results.get('datasets', {}).items():
        print(f"  {dataset_name}:")
        print(f"    Samples: {dataset_info['total_samples']}")
        print(f"    Categories: {len(dataset_info.get('categories', []))}")

    # Print evaluation status
    print("\n✅ Evaluations:")
    for technique, eval_info in results.get('evaluations', {}).items():
        status = eval_info.get('status', 'unknown')
        print(f"  {technique}: {status}")

    # Print metrics summary
    if 'metrics' in results:
        print("\n📈 Metrics:")
        metrics = results['metrics']
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, dict):
                print(f"  {metric_name}:")
                for tech, value in metric_values.items():
                    print(f"    {tech}: {value}")

    # Print metadata
    print("\n⏱️  Metadata:")
    metadata = results.get('metadata', {})
    print(f"  Total samples processed: {metadata.get('total_samples', 0)}")
    print(f"  Total API calls: {metadata.get('total_api_calls', 0)}")
    if metadata.get('start_time'):
        print(f"  Start time: {metadata['start_time']}")
    if metadata.get('end_time'):
        print(f"  End time: {metadata['end_time']}")

    # Print visualizations info
    if 'visualizations' in results:
        print("\n📊 Visualizations:")
        viz = results['visualizations']
        print(f"  Report: {viz.get('report_path', 'N/A')}")
        print(f"  Figures: {viz.get('figures_dir', 'N/A')}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze experimental results")
    parser.add_argument(
        'results',
        nargs='?',
        default='results/experiment_results.json',
        help='Path to results JSON file'
    )

    args = parser.parse_args()

    if not Path(args.results).exists():
        print(f"❌ Results file not found: {args.results}")
        return 1

    analyze_results(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
