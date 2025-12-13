"""
Enhanced Visualization Module

Generates 4 key visualizations for prompt optimization experiments:
1. improvement_over_baseline.png - Bar chart of accuracy improvements
2. accuracy_comparison_full.png - Grouped bar chart (accuracy, loss, entropy)
3. top_mistakes.png - Confusion-type chart for mistakes
4. metric_trends.png - Line plot of normalized metrics
"""

from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def generate_all_visualizations(
    results: Dict[str, Any],
    output_dir: str = "results/figures"
) -> str:
    """
    Generate all 4 required visualizations.

    Args:
        results: Experimental results dictionary
        output_dir: Directory to save figures

    Returns:
        Path to figures directory
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    techniques = results.get("techniques", {})
    baseline_name = results.get("baseline_technique", "baseline")

    if not techniques:
        print("    ⚠️  No technique results to visualize")
        return str(output_path)

    # Get baseline accuracy
    baseline_acc = 0.0
    if baseline_name in techniques:
        baseline_acc = techniques[baseline_name].get("metrics", {}).get("accuracy", 0.0)

    # Generate each visualization
    try:
        # 1. Improvement over baseline
        plot_improvement_over_baseline(results, str(output_path / "improvement_over_baseline.png"))
        print("    ✓ improvement_over_baseline.png")
    except Exception as e:
        print(f"    ⚠️  Error generating improvement plot: {e}")

    try:
        # 2. Accuracy comparison (full metrics)
        plot_accuracy_comparison_full(results, str(output_path / "accuracy_comparison_full.png"))
        print("    ✓ accuracy_comparison_full.png")
    except Exception as e:
        print(f"    ⚠️  Error generating accuracy comparison: {e}")

    try:
        # 3. Top mistakes (if mistakes exist)
        has_mistakes = any(
            not pred.get("correct", False)
            for tech_data in techniques.values()
            for pred in tech_data.get("predictions", [])
        )

        if has_mistakes:
            plot_top_mistakes(results, str(output_path / "top_mistakes.png"))
            print("    ✓ top_mistakes.png")
        else:
            print("    ⏭️  top_mistakes.png (no mistakes to visualize)")
    except Exception as e:
        print(f"    ⚠️  Error generating top mistakes: {e}")

    try:
        # 4. Metric trends
        plot_metric_trends(results, str(output_path / "metric_trends.png"))
        print("    ✓ metric_trends.png")
    except Exception as e:
        print(f"    ⚠️  Error generating metric trends: {e}")

    return str(output_path)


def plot_improvement_over_baseline(results: Dict[str, Any], save_path: str):
    """
    Plot 1: Bar chart showing Δ accuracy for each technique.

    Args:
        results: Experimental results
        save_path: Path to save figure
    """
    techniques = results.get("techniques", {})
    baseline_name = results.get("baseline_technique", "baseline")

    # Get baseline accuracy
    baseline_acc = 0.0
    if baseline_name in techniques:
        baseline_acc = techniques[baseline_name].get("metrics", {}).get("accuracy", 0.0)

    # Calculate improvements
    improvements = {}
    for tech_name, tech_data in techniques.items():
        if tech_name == baseline_name:
            continue
        acc = tech_data.get("metrics", {}).get("accuracy", 0.0)
        delta = (acc - baseline_acc) * 100  # Percentage points
        improvements[tech_name] = delta

    if not improvements:
        # Just baseline - create empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No techniques to compare (only baseline)",
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    # Sort by improvement
    sorted_improvements = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    tech_names = [_format_technique_name(t[0]) for t in sorted_improvements]
    deltas = [t[1] for t in sorted_improvements]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color bars based on positive/negative
    colors = ['green' if d > 0 else 'red' for d in deltas]
    bars = ax.bar(tech_names, deltas, color=colors, alpha=0.7)

    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        label_y = height + (1 if height > 0 else -1)
        ax.text(bar.get_x() + bar.get_width()/2., label_y,
                f'{delta:+.1f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

    # Add zero line
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Improvement Over Baseline (Baseline: {baseline_acc*100:.1f}%)',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_accuracy_comparison_full(results: Dict[str, Any], save_path: str):
    """
    Plot 2: Grouped bar chart showing accuracy, loss, and entropy per technique.

    Args:
        results: Experimental results
        save_path: Path to save figure
    """
    techniques = results.get("techniques", {})

    if not techniques:
        return

    # Extract data
    tech_names = []
    accuracies = []
    losses = []
    entropies = []

    # Sort by accuracy
    sorted_techs = sorted(
        techniques.items(),
        key=lambda x: x[1].get("metrics", {}).get("accuracy", 0.0),
        reverse=True
    )

    for tech_name, tech_data in sorted_techs:
        metrics = tech_data.get("metrics", {})
        tech_names.append(_format_technique_name(tech_name))
        accuracies.append(metrics.get("accuracy", 0.0) * 100)  # Convert to percentage
        losses.append(metrics.get("loss", 0.0))
        entropies.append(metrics.get("entropy", 0.0))

    # Create grouped bar chart
    x = np.arange(len(tech_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))

    # Normalize metrics to 0-100 scale for visual comparison
    # Accuracy is already in %
    # Loss: invert and scale (lower is better)
    max_loss = max(losses) if losses else 1.0
    normalized_losses = [(1 - l/max_loss) * 100 if max_loss > 0 else 0 for l in losses]

    # Entropy: scale to 0-100
    max_entropy = max(entropies) if entropies else 1.0
    normalized_entropies = [(e/max_entropy) * 100 if max_entropy > 0 else 0 for e in entropies]

    # Plot bars
    bars1 = ax.bar(x - width, accuracies, width, label='Accuracy (%)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, normalized_losses, width, label='Loss (inverted, normalized)', color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, normalized_entropies, width, label='Entropy (normalized)', color='seagreen', alpha=0.8)

    # Add value labels on accuracy bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0-100)', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Metric Comparison Across Techniques', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tech_names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_top_mistakes(results: Dict[str, Any], save_path: str, top_k: int = 10):
    """
    Plot 3: Confusion-type chart showing top mistakes.

    Args:
        results: Experimental results
        save_path: Path to save figure
        top_k: Number of top mistakes to show
    """
    techniques = results.get("techniques", {})

    # Collect all mistakes
    mistakes = []
    for tech_name, tech_data in techniques.items():
        predictions = tech_data.get("predictions", [])
        for pred in predictions:
            if not pred.get("correct", False):
                mistakes.append({
                    "technique": tech_name,
                    "sample_id": pred.get("sample_id", "N/A"),
                    "question": pred.get("question", "N/A")[:50] + "...",
                    "category": pred.get("category", "unknown"),
                })

    if not mistakes:
        return

    # Count mistakes by technique and category
    from collections import defaultdict
    mistake_counts = defaultdict(lambda: defaultdict(int))

    for mistake in mistakes[:top_k * 5]:  # Consider more mistakes for aggregation
        tech = _format_technique_name(mistake["technique"])
        category = mistake["category"]
        mistake_counts[tech][category] += 1

    # Create heatmap data
    all_techniques = sorted(mistake_counts.keys())
    all_categories = sorted(set(
        cat for tech_cats in mistake_counts.values() for cat in tech_cats.keys()
    ))

    # Build matrix
    matrix = []
    for tech in all_techniques:
        row = [mistake_counts[tech][cat] for cat in all_categories]
        matrix.append(row)

    if not matrix or not all_categories:
        return

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(all_techniques) * 0.5)))

    sns.heatmap(
        matrix,
        annot=True,
        fmt='d',
        cmap='YlOrRd',
        xticklabels=all_categories,
        yticklabels=all_techniques,
        cbar_kws={'label': 'Number of Mistakes'},
        ax=ax
    )

    ax.set_xlabel('Error Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Technique', fontsize=12, fontweight='bold')
    ax.set_title(f'Mistake Distribution by Technique and Category (Top {len(mistakes)} mistakes)',
                 fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_trends(results: Dict[str, Any], save_path: str):
    """
    Plot 4: Line plot comparing normalized metrics (accuracy, loss, entropy) across techniques.

    Args:
        results: Experimental results
        save_path: Path to save figure
    """
    techniques = results.get("techniques", {})

    if not techniques:
        return

    # Extract and normalize metrics
    tech_names = []
    accuracies = []
    losses = []
    entropies = []

    # Sort by accuracy for consistent ordering
    sorted_techs = sorted(
        techniques.items(),
        key=lambda x: x[1].get("metrics", {}).get("accuracy", 0.0),
        reverse=True
    )

    for tech_name, tech_data in sorted_techs:
        metrics = tech_data.get("metrics", {})
        tech_names.append(_format_technique_name(tech_name))
        accuracies.append(metrics.get("accuracy", 0.0))
        losses.append(metrics.get("loss", 0.0))
        entropies.append(metrics.get("entropy", 0.0))

    # Normalize to 0-1 scale
    def normalize(values):
        """Normalize values to 0-1 range."""
        if not values:
            return values
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]

    # Accuracy already 0-1
    norm_accuracies = accuracies

    # Loss: invert (lower is better)
    if losses:
        max_loss = max(losses)
        min_loss = min(losses)
        if max_loss > min_loss:
            norm_losses = [(max_loss - l) / (max_loss - min_loss) for l in losses]
        else:
            norm_losses = [0.5] * len(losses)
    else:
        norm_losses = [0] * len(tech_names)

    # Entropy: normalize
    norm_entropies = normalize(entropies)

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = range(len(tech_names))

    ax.plot(x, norm_accuracies, marker='o', linewidth=2, markersize=8,
            label='Accuracy (higher is better)', color='steelblue')
    ax.plot(x, norm_losses, marker='s', linewidth=2, markersize=8,
            label='Loss (inverted, normalized)', color='coral')
    ax.plot(x, norm_entropies, marker='^', linewidth=2, markersize=8,
            label='Entropy (normalized)', color='seagreen')

    # Add value labels for accuracy
    for i, (acc, loss, ent) in enumerate(zip(norm_accuracies, norm_losses, norm_entropies)):
        ax.text(i, acc + 0.02, f'{acc:.2f}', ha='center', va='bottom', fontsize=8, color='steelblue')

    ax.set_xlabel('Prompt Technique', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Metric Trends Across Techniques (Normalized)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(tech_names, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def _format_technique_name(tech_name: str) -> str:
    """
    Format technique name for display.

    Args:
        tech_name: Raw technique name

    Returns:
        Formatted name
    """
    # Replace underscores with spaces and title case
    formatted = tech_name.replace("_", " ").title()

    # Handle special cases
    replacements = {
        "Chain Of Thought": "Chain-of-Thought",
        "Chain Of Thought Plus Plus": "Chain-of-Thought++",
        "React": "ReAct",
        "Tree Of Thoughts": "Tree-of-Thoughts",
    }

    for old, new in replacements.items():
        formatted = formatted.replace(old, new)

    return formatted
