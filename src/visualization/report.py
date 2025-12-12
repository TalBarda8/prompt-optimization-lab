"""
Visualization Report Generator

Generates comprehensive visualization reports from experimental results.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .plotters import (
    plot_accuracy_comparison,
    plot_loss_comparison,
    plot_entropy_distribution,
    plot_perplexity_distribution,
    plot_response_length_distribution,
    plot_performance_heatmap,
    plot_significance_matrix,
    plot_category_accuracy,
    plot_confidence_intervals,
    plot_time_series_performance,
    plot_correlation_matrix,
    plot_technique_rankings,
)


def save_all_plots(
    results: Dict[str, Any],
    output_dir: str = "results/figures",
    formats: List[str] = None,
) -> Dict[str, List[str]]:
    """
    Generate and save all 12 visualizations.

    Args:
        results: Dictionary containing all experimental results
        output_dir: Directory to save figures
        formats: List of formats to save (default: ['png', 'pdf'])

    Returns:
        Dictionary mapping plot names to saved file paths
    """
    if formats is None:
        formats = ['png', 'pdf']

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Extract data from results
    techniques = results.get('techniques', [])

    # 1. Accuracy Comparison
    if 'accuracy' in results:
        for fmt in formats:
            path = output_path / f"01_accuracy_comparison.{fmt}"
            plot_accuracy_comparison(results['accuracy'], save_path=str(path))
            plt.close()
            saved_files.setdefault('accuracy_comparison', []).append(str(path))

    # 2. Loss Comparison
    if 'loss' in results:
        for fmt in formats:
            path = output_path / f"02_loss_comparison.{fmt}"
            plot_loss_comparison(results['loss'], save_path=str(path))
            plt.close()
            saved_files.setdefault('loss_comparison', []).append(str(path))

    # 3. Entropy Distribution
    if 'entropy_dist' in results:
        for fmt in formats:
            path = output_path / f"03_entropy_distribution.{fmt}"
            plot_entropy_distribution(results['entropy_dist'], save_path=str(path))
            plt.close()
            saved_files.setdefault('entropy_distribution', []).append(str(path))

    # 4. Perplexity Distribution
    if 'perplexity_dist' in results:
        for fmt in formats:
            path = output_path / f"04_perplexity_distribution.{fmt}"
            plot_perplexity_distribution(results['perplexity_dist'], save_path=str(path))
            plt.close()
            saved_files.setdefault('perplexity_distribution', []).append(str(path))

    # 5. Response Length Distribution
    if 'length_dist' in results:
        for fmt in formats:
            path = output_path / f"05_response_length_distribution.{fmt}"
            plot_response_length_distribution(results['length_dist'], save_path=str(path))
            plt.close()
            saved_files.setdefault('length_distribution', []).append(str(path))

    # 6. Performance Heatmap
    if 'performance_matrix' in results:
        for fmt in formats:
            path = output_path / f"06_performance_heatmap.{fmt}"
            plot_performance_heatmap(results['performance_matrix'], save_path=str(path))
            plt.close()
            saved_files.setdefault('performance_heatmap', []).append(str(path))

    # 7. Significance Matrix
    if 'p_values' in results:
        for fmt in formats:
            path = output_path / f"07_significance_matrix.{fmt}"
            plot_significance_matrix(results['p_values'], save_path=str(path))
            plt.close()
            saved_files.setdefault('significance_matrix', []).append(str(path))

    # 8. Category Accuracy
    if 'category_accuracy' in results:
        for fmt in formats:
            path = output_path / f"08_category_accuracy.{fmt}"
            plot_category_accuracy(results['category_accuracy'], save_path=str(path))
            plt.close()
            saved_files.setdefault('category_accuracy', []).append(str(path))

    # 9. Confidence Intervals
    if 'confidence_intervals' in results:
        ci = results['confidence_intervals']
        for fmt in formats:
            path = output_path / f"09_confidence_intervals.{fmt}"
            plot_confidence_intervals(
                ci['means'], ci['lower'], ci['upper'],
                save_path=str(path)
            )
            plt.close()
            saved_files.setdefault('confidence_intervals', []).append(str(path))

    # 10. Time Series Performance
    if 'time_series' in results:
        for fmt in formats:
            path = output_path / f"10_time_series_performance.{fmt}"
            plot_time_series_performance(results['time_series'], save_path=str(path))
            plt.close()
            saved_files.setdefault('time_series', []).append(str(path))

    # 11. Correlation Matrix
    if 'metrics_df' in results:
        for fmt in formats:
            path = output_path / f"11_correlation_matrix.{fmt}"
            plot_correlation_matrix(results['metrics_df'], save_path=str(path))
            plt.close()
            saved_files.setdefault('correlation_matrix', []).append(str(path))

    # 12. Technique Rankings
    if 'rankings' in results and 'scores' in results:
        for fmt in formats:
            path = output_path / f"12_technique_rankings.{fmt}"
            plot_technique_rankings(
                results['rankings'], results['scores'],
                save_path=str(path)
            )
            plt.close()
            saved_files.setdefault('technique_rankings', []).append(str(path))

    return saved_files


def generate_visualization_report(
    results: Dict[str, Any],
    output_dir: str = "results",
    include_plots: bool = True,
) -> str:
    """
    Generate comprehensive visualization report.

    Args:
        results: Experimental results dictionary
        output_dir: Directory for outputs
        include_plots: Whether to generate plot files

    Returns:
        Path to report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save all plots if requested
    saved_files = {}
    if include_plots:
        saved_files = save_all_plots(results, output_dir=str(output_path / "figures"))

    # Generate markdown report
    report_lines = [
        "# Prompt Optimization Visualization Report",
        "",
        "## Overview",
        "",
        f"Total techniques evaluated: {len(results.get('techniques', []))}",
        f"Total samples: {results.get('total_samples', 'N/A')}",
        "",
        "## Visualizations Generated",
        "",
    ]

    # List all generated plots
    for i, (plot_name, files) in enumerate(saved_files.items(), 1):
        report_lines.append(f"{i}. **{plot_name.replace('_', ' ').title()}**")
        for file_path in files:
            report_lines.append(f"   - `{Path(file_path).name}`")
        report_lines.append("")

    # Add summary statistics if available
    if 'accuracy' in results:
        report_lines.extend([
            "## Summary Statistics",
            "",
            "### Accuracy by Technique",
            "",
        ])

        for technique, acc in sorted(results['accuracy'].items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"- **{technique}**: {acc:.4f}")

        report_lines.append("")

    # Save report
    report_path = output_path / "visualization_report.md"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return str(report_path)
