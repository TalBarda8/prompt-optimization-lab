"""
Visualization Plotters

Implements 12 required visualizations from PRD Section 4.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_accuracy_comparison(
    results: Dict[str, float],
    title: str = "Accuracy Comparison by Prompt Technique",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 1: Accuracy comparison across techniques (bar chart).

    Args:
        results: Dict mapping technique names to accuracy scores
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    techniques = list(results.keys())
    accuracies = list(results.values())

    # Create bar chart
    bars = ax.bar(techniques, accuracies, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_loss_comparison(
    results: Dict[str, float],
    title: str = "Loss Function Comparison by Prompt Technique",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 2: Loss function comparison (bar chart).

    Args:
        results: Dict mapping technique names to loss values
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    techniques = list(results.keys())
    losses = list(results.values())

    # Create bar chart (lower is better, so use reversed colormap)
    bars = ax.bar(techniques, losses, color='coral', alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_entropy_distribution(
    data: Dict[str, List[float]],
    title: str = "Entropy Distribution by Prompt Technique",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 3: Entropy distribution (box plots).

    Args:
        data: Dict mapping technique names to lists of entropy values
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plot
    techniques = list(data.keys())
    entropy_data = [data[t] for t in techniques]

    # Create box plot
    bp = ax.boxplot(entropy_data, tick_labels=techniques, patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entropy (bits)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_perplexity_distribution(
    data: Dict[str, List[float]],
    title: str = "Perplexity Distribution by Prompt Technique",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 4: Perplexity distribution (box plots).

    Args:
        data: Dict mapping technique names to lists of perplexity values
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    techniques = list(data.keys())
    perplexity_data = [data[t] for t in techniques]

    # Create box plot
    bp = ax.boxplot(perplexity_data, tick_labels=techniques, patch_artist=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.7)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_response_length_distribution(
    data: Dict[str, List[int]],
    title: str = "Response Length Distribution by Prompt Technique",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 5: Response length distribution (violin plots).

    Args:
        data: Dict mapping technique names to lists of token counts
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for violin plot
    df_data = []
    for technique, lengths in data.items():
        for length in lengths:
            df_data.append({'Technique': technique, 'Length': length})

    df = pd.DataFrame(df_data)

    # Create violin plot
    sns.violinplot(data=df, x='Technique', y='Length', hue='Technique', ax=ax, palette='Set2', legend=False)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Response Length (tokens)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_performance_heatmap(
    data: pd.DataFrame,
    title: str = "Technique Performance Heatmap",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 6: Performance heatmap (techniques × metrics).

    Args:
        data: DataFrame with techniques as rows, metrics as columns
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn',
                center=0.5, ax=ax, cbar_kws={'label': 'Score'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Technique', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_significance_matrix(
    p_values: pd.DataFrame,
    title: str = "Statistical Significance Matrix (p-values)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 7: Statistical significance matrix.

    Args:
        p_values: DataFrame of p-values (techniques × techniques)
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(p_values, dtype=bool))

    # Create heatmap with significance thresholds
    sns.heatmap(p_values, mask=mask, annot=True, fmt='.4f',
                cmap='RdYlGn_r', center=0.05, vmin=0, vmax=0.1,
                ax=ax, cbar_kws={'label': 'p-value'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Technique', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_category_accuracy(
    data: Dict[str, Dict[str, float]],
    title: str = "Category-wise Accuracy Breakdown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 8: Category-wise accuracy breakdown.

    Args:
        data: Nested dict {technique: {category: accuracy}}
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Convert to DataFrame
    df = pd.DataFrame(data).T

    # Create grouped bar chart
    df.plot(kind='bar', ax=ax, width=0.8, colormap='tab10')

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confidence_intervals(
    means: Dict[str, float],
    ci_lower: Dict[str, float],
    ci_upper: Dict[str, float],
    title: str = "Accuracy with 95% Confidence Intervals",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 9: Confidence intervals with error bars.

    Args:
        means: Dict of mean accuracy per technique
        ci_lower: Dict of lower CI bounds
        ci_upper: Dict of upper CI bounds
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    techniques = list(means.keys())
    mean_vals = list(means.values())
    lower_vals = [means[t] - ci_lower[t] for t in techniques]
    upper_vals = [ci_upper[t] - means[t] for t in techniques]

    # Create error bar chart
    ax.errorbar(techniques, mean_vals,
                yerr=[lower_vals, upper_vals],
                fmt='o', markersize=8, capsize=10,
                color='steelblue', ecolor='gray', alpha=0.8)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_time_series_performance(
    data: Dict[str, List[Tuple[int, float]]],
    title: str = "Performance Over Time",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 10: Time-series performance (if applicable).

    Args:
        data: Dict mapping technique to list of (time, metric) tuples
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for technique, points in data.items():
        if points:
            times, metrics = zip(*points)
            ax.plot(times, metrics, marker='o', label=technique, linewidth=2)

    ax.set_xlabel('Sample Index / Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance Metric', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_correlation_matrix(
    data: pd.DataFrame,
    title: str = "Metric Correlation Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 11: Correlation matrix of metrics.

    Args:
        data: DataFrame with metrics as columns
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate correlation matrix
    corr = data.corr()

    # Create heatmap
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={'label': 'Correlation'})

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_technique_rankings(
    rankings: Dict[str, int],
    scores: Dict[str, float],
    title: str = "Prompt Technique Rankings",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot 12: Overall technique rankings.

    Args:
        rankings: Dict mapping technique to rank (1=best)
        scores: Dict mapping technique to composite score
        title: Plot title
        save_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by rank
    sorted_techniques = sorted(rankings.keys(), key=lambda x: rankings[x])
    sorted_scores = [scores[t] for t in sorted_techniques]
    sorted_ranks = [rankings[t] for t in sorted_techniques]

    # Create horizontal bar chart
    y_pos = np.arange(len(sorted_techniques))
    bars = ax.barh(y_pos, sorted_scores, color='teal', alpha=0.8)

    # Add rank labels
    for i, (bar, rank) in enumerate(zip(bars, sorted_ranks)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f' Rank #{rank} ({width:.3f})',
                ha='left', va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_techniques)
    ax.set_xlabel('Composite Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Best at top
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
