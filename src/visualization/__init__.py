"""
Visualization Module

Implements 12 required visualizations from PRD Section 4.
Enhanced with 4 key visualizations for experiment results.
"""

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

from .report import generate_visualization_report, save_all_plots

# Import enhanced visualization functions
from .visualization import (
    generate_all_visualizations,
    plot_improvement_over_baseline,
    plot_accuracy_comparison_full,
    plot_top_mistakes,
    plot_metric_trends,
)

__all__ = [
    # Individual plotters (legacy)
    "plot_accuracy_comparison",
    "plot_loss_comparison",
    "plot_entropy_distribution",
    "plot_perplexity_distribution",
    "plot_response_length_distribution",
    "plot_performance_heatmap",
    "plot_significance_matrix",
    "plot_category_accuracy",
    "plot_confidence_intervals",
    "plot_time_series_performance",
    "plot_correlation_matrix",
    "plot_technique_rankings",
    # Report generation (legacy)
    "generate_visualization_report",
    "save_all_plots",
    # Enhanced visualizations (new)
    "generate_all_visualizations",
    "plot_improvement_over_baseline",
    "plot_accuracy_comparison_full",
    "plot_top_mistakes",
    "plot_metric_trends",
]
