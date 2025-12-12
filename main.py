#!/usr/bin/env python3
"""
Prompt Optimization Lab - Main CLI

Command-line interface for running prompt optimization experiments.
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import ExperimentConfig, ExperimentOrchestrator
from data import create_dataset_a, create_dataset_b, save_dataset


def cmd_create_datasets(args):
    """Create datasets command."""
    print("Creating datasets...")

    # Create Dataset A
    print("\nCreating Dataset A (Simple QA)...")
    dataset_a = create_dataset_a()
    save_dataset(dataset_a, "data/dataset_a.json")
    print(f"✓ Dataset A saved: {dataset_a['total_samples']} samples")

    # Create Dataset B
    print("\nCreating Dataset B (Multi-step Reasoning)...")
    dataset_b = create_dataset_b()
    save_dataset(dataset_b, "data/dataset_b.json")
    print(f"✓ Dataset B saved: {dataset_b['total_samples']} samples")

    print(f"\n✅ Total: {dataset_a['total_samples'] + dataset_b['total_samples']} samples created")


def cmd_run_experiment(args):
    """Run full experiment command."""
    print("Running full experiment pipeline...")

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ExperimentConfig(**config_dict)
    else:
        # Default configuration
        config = ExperimentConfig(
            dataset_paths={
                "dataset_a": args.dataset_a or "data/dataset_a.json",
                "dataset_b": args.dataset_b or "data/dataset_b.json",
            },
            llm_provider=args.provider,
            llm_model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            techniques=args.techniques or [
                "baseline",
                "chain_of_thought",
                "chain_of_thought_plus_plus",
                "react",
                "tree_of_thoughts",
                "role_based",
                "few_shot",
            ],
            output_dir=args.output,
        )

    # Create and run orchestrator
    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_full_pipeline()

    print("\n✅ Experiment complete!")
    print(f"Results: {config.output_dir}/experiment_results.json")
    print(f"Visualizations: {config.output_dir}/figures/")

    return results


def cmd_run_baseline(args):
    """Run baseline only command."""
    print("Running baseline evaluation...")

    config = ExperimentConfig(
        dataset_paths={
            "dataset_a": args.dataset_a or "data/dataset_a.json",
        },
        llm_provider=args.provider,
        llm_model=args.model,
        techniques=["baseline"],
        output_dir=args.output,
    )

    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_full_pipeline()

    print("\n✅ Baseline evaluation complete!")


def cmd_compare_techniques(args):
    """Compare specific techniques command."""
    print(f"Comparing techniques: {', '.join(args.techniques)}")

    config = ExperimentConfig(
        dataset_paths={
            "dataset_a": args.dataset_a or "data/dataset_a.json",
            "dataset_b": args.dataset_b or "data/dataset_b.json",
        },
        llm_provider=args.provider,
        llm_model=args.model,
        techniques=args.techniques,
        output_dir=args.output,
    )

    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_full_pipeline()

    print("\n✅ Technique comparison complete!")


def cmd_visualize(args):
    """Generate visualizations from existing results."""
    from visualization import generate_visualization_report

    print(f"Generating visualizations from: {args.results}")

    with open(args.results, 'r') as f:
        results = json.load(f)

    report_path = generate_visualization_report(
        results,
        output_dir=args.output,
        include_plots=True,
    )

    print(f"\n✅ Visualizations generated!")
    print(f"Report: {report_path}")


def cmd_validate_datasets(args):
    """Validate datasets command."""
    from data import load_dataset, validate_dataset

    print("Validating datasets...")

    datasets = [
        ("Dataset A", args.dataset_a or "data/dataset_a.json"),
        ("Dataset B", args.dataset_b or "data/dataset_b.json"),
    ]

    for name, path in datasets:
        if not Path(path).exists():
            print(f"⚠️  {name} not found: {path}")
            continue

        print(f"\nValidating {name}...")
        dataset = load_dataset(path)
        report = validate_dataset(dataset)

        if report["valid"]:
            print(f"✅ {name} is valid!")
            print(f"   Samples: {report['total_samples']}")
            print(f"   Categories: {report['statistics']['category_distribution']}")
        else:
            print(f"❌ {name} has errors:")
            for error in report["errors"][:5]:  # Show first 5
                print(f"   - {error}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prompt Optimization Lab - Experimental Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create datasets
  python main.py create-datasets

  # Run full experiment
  python main.py run-experiment --model gpt-4

  # Run baseline only
  python main.py run-baseline

  # Compare specific techniques
  python main.py compare --techniques baseline chain_of_thought

  # Validate datasets
  python main.py validate

  # Generate visualizations from existing results
  python main.py visualize --results results/experiment_results.json
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Create datasets command
    parser_create = subparsers.add_parser(
        'create-datasets',
        help='Create Dataset A and Dataset B'
    )
    parser_create.set_defaults(func=cmd_create_datasets)

    # Run experiment command
    parser_run = subparsers.add_parser(
        'run-experiment',
        help='Run full experimental pipeline'
    )
    parser_run.add_argument('--config', type=str, help='Path to config JSON file')
    parser_run.add_argument('--dataset-a', type=str, help='Path to Dataset A')
    parser_run.add_argument('--dataset-b', type=str, help='Path to Dataset B')
    parser_run.add_argument('--provider', default='openai', choices=['openai', 'anthropic'])
    parser_run.add_argument('--model', default='gpt-4', help='Model name')
    parser_run.add_argument('--temperature', type=float, default=0.0)
    parser_run.add_argument('--max-tokens', type=int, default=500)
    parser_run.add_argument('--techniques', nargs='+', help='Techniques to evaluate')
    parser_run.add_argument('--output', default='results', help='Output directory')
    parser_run.set_defaults(func=cmd_run_experiment)

    # Run baseline command
    parser_baseline = subparsers.add_parser(
        'run-baseline',
        help='Run baseline evaluation only'
    )
    parser_baseline.add_argument('--dataset-a', type=str, help='Path to Dataset A')
    parser_baseline.add_argument('--provider', default='openai')
    parser_baseline.add_argument('--model', default='gpt-4')
    parser_baseline.add_argument('--output', default='results')
    parser_baseline.set_defaults(func=cmd_run_baseline)

    # Compare techniques command
    parser_compare = subparsers.add_parser(
        'compare',
        help='Compare specific techniques'
    )
    parser_compare.add_argument('--techniques', nargs='+', required=True)
    parser_compare.add_argument('--dataset-a', type=str)
    parser_compare.add_argument('--dataset-b', type=str)
    parser_compare.add_argument('--provider', default='openai')
    parser_compare.add_argument('--model', default='gpt-4')
    parser_compare.add_argument('--output', default='results')
    parser_compare.set_defaults(func=cmd_compare_techniques)

    # Visualize command
    parser_viz = subparsers.add_parser(
        'visualize',
        help='Generate visualizations from results'
    )
    parser_viz.add_argument('--results', required=True, help='Path to results JSON')
    parser_viz.add_argument('--output', default='results', help='Output directory')
    parser_viz.set_defaults(func=cmd_visualize)

    # Validate command
    parser_validate = subparsers.add_parser(
        'validate',
        help='Validate datasets'
    )
    parser_validate.add_argument('--dataset-a', type=str)
    parser_validate.add_argument('--dataset-b', type=str)
    parser_validate.set_defaults(func=cmd_validate_datasets)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Execute command
    try:
        args.func(args)
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
