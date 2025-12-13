"""
Experiment Summary Generator

Generates human-readable summaries of experimental results for CLI output.
Enhanced with Rich tables and ASCII bar charts for better visualization.
"""

from typing import Dict, List, Any, Optional
import numpy as np

# Try to import Rich for enhanced terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def print_experiment_summary(
    results: Dict[str, Any],
    model_name: str = "LLM",
    show_top_mistakes: bool = True,
    top_k: int = 5,
):
    """
    Print comprehensive experiment summary to console.
    Enhanced with Rich tables and ASCII bar charts.

    Args:
        results: Experimental results dictionary
        model_name: Name of the model used
        show_top_mistakes: Whether to show top mistakes
        top_k: Number of top mistakes to show
    """
    print("\n" + "=" * 80)
    print(f" EXPERIMENT SUMMARY ({model_name})")
    print("=" * 80)

    # Extract technique results
    techniques = results.get("techniques", {})
    baseline_name = results.get("baseline_technique", "baseline")

    if not techniques:
        print("\n⚠️  No results available")
        return

    # Get baseline accuracy
    baseline_acc = 0.0
    if baseline_name in techniques:
        baseline_acc = techniques[baseline_name].get("metrics", {}).get("accuracy", 0.0)

    # Sort techniques by accuracy
    sorted_techniques = sorted(
        techniques.items(),
        key=lambda x: x[1].get("metrics", {}).get("accuracy", 0.0),
        reverse=True
    )

    # ========================================
    # 1. RICH TABLE (or ASCII fallback)
    # ========================================
    print("\n📊 ACCURACY COMPARISON TABLE:")
    print("-" * 80)

    if RICH_AVAILABLE:
        _print_rich_accuracy_table(sorted_techniques, baseline_name, baseline_acc)
    else:
        _print_ascii_accuracy_table(sorted_techniques, baseline_name, baseline_acc)

    # ========================================
    # 2. ASCII BAR CHART
    # ========================================
    print("\n📊 ACCURACY BAR CHART:")
    print("-" * 80)
    _print_ascii_bar_chart(sorted_techniques)

    # ========================================
    # 3. IMPROVEMENT OVER BASELINE
    # ========================================
    print("\n📈 IMPROVEMENT OVER BASELINE:")
    print("-" * 80)
    _print_improvements(sorted_techniques, baseline_name, baseline_acc)

    # Find best technique
    if sorted_techniques:
        best_tech_name, best_tech_data = sorted_techniques[0]
        best_acc = best_tech_data.get("metrics", {}).get("accuracy", 0.0)
        best_improvement = (best_acc - baseline_acc) * 100

        print("\n" + "=" * 80)
        print(f"🏆 BEST TECHNIQUE: {best_tech_name.replace('_', ' ').title()}")
        print(f"   Accuracy: {best_acc*100:.1f}%")
        if best_tech_name != baseline_name:
            print(f"   Improvement over baseline: +{best_improvement:.1f}%")
        print("=" * 80)

    # Print loss comparison
    print("\n📉 LOSS FUNCTION VALUES (lower is better):")
    print("-" * 80)

    sorted_by_loss = sorted(
        techniques.items(),
        key=lambda x: x[1].get("metrics", {}).get("loss", float('inf'))
    )

    for tech_name, tech_data in sorted_by_loss:
        metrics = tech_data.get("metrics", {})
        loss = metrics.get("loss", 0.0)
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."
        print(f"  {display_name:32} ... {loss:.4f}")

    # Print entropy/perplexity stats
    print("\n🔍 INFORMATION-THEORETIC METRICS:")
    print("-" * 80)
    print(f"  {'Technique':32}  {'Entropy (bits)':>15}  {'Perplexity':>12}")
    print("  " + "-" * 62)

    for tech_name, tech_data in sorted_techniques:
        metrics = tech_data.get("metrics", {})
        entropy = metrics.get("entropy", 0.0)
        perplexity = metrics.get("perplexity", 0.0)
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        is_estimated = metrics.get("metrics_estimated", False)
        suffix = " (est.)" if is_estimated else ""

        print(f"  {display_name:32}  {entropy:>13.2f}{suffix:>2}  {perplexity:>12.2f}")

    # Show top mistakes if requested
    if show_top_mistakes:
        print_top_mistakes(results, top_k=top_k)

    # Print statistical significance if available
    if "statistical_tests" in results:
        print_statistical_significance(results["statistical_tests"])

    print("\n" + "=" * 80)
    print("✅ Experiment complete!")
    print("=" * 80 + "\n")


def _print_rich_accuracy_table(sorted_techniques, baseline_name, baseline_acc):
    """
    Print Rich table with accuracy comparison.

    Args:
        sorted_techniques: List of (technique_name, technique_data) sorted by accuracy
        baseline_name: Name of baseline technique
        baseline_acc: Baseline accuracy value
    """
    console = Console()
    table = Table(show_header=True, header_style="bold cyan")

    # Add columns
    table.add_column("Technique", style="white", width=30)
    table.add_column("Accuracy (%)", justify="right", style="yellow")
    table.add_column("Δ from Baseline", justify="right", style="white")
    table.add_column("Loss", justify="right", style="magenta")
    table.add_column("Entropy", justify="right", style="green")

    # Add rows
    for tech_name, tech_data in sorted_techniques:
        metrics = tech_data.get("metrics", {})
        acc = metrics.get("accuracy", 0.0)
        loss = metrics.get("loss", 0.0)
        entropy = metrics.get("entropy", 0.0)
        is_estimated = metrics.get("metrics_estimated", False)

        # Format technique name
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."

        # Calculate improvement
        delta = (acc - baseline_acc) * 100

        # Color code delta
        if tech_name == baseline_name:
            delta_str = "(baseline)"
            delta_style = "dim"
        elif delta > 0:
            delta_str = f"+{delta:.1f}%"
            delta_style = "bold green"
        elif delta < 0:
            delta_str = f"{delta:.1f}%"
            delta_style = "bold red"
        else:
            delta_str = "0.0%"
            delta_style = "dim"

        # Entropy suffix
        entropy_str = f"{entropy:.2f}"
        if is_estimated:
            entropy_str += " (est.)"

        # Add row
        table.add_row(
            display_name,
            f"{acc*100:.1f}",
            Text(delta_str, style=delta_style),
            f"{loss:.4f}",
            entropy_str
        )

    console.print(table)


def _print_ascii_accuracy_table(sorted_techniques, baseline_name, baseline_acc):
    """
    Print ASCII fallback table with accuracy comparison.

    Args:
        sorted_techniques: List of (technique_name, technique_data) sorted by accuracy
        baseline_name: Name of baseline technique
        baseline_acc: Baseline accuracy value
    """
    # Print header
    print(f"  {'Technique':30} | {'Accuracy':>10} | {'Δ Baseline':>12} | {'Loss':>8} | {'Entropy':>10}")
    print("  " + "-" * 80)

    # Print rows
    for tech_name, tech_data in sorted_techniques:
        metrics = tech_data.get("metrics", {})
        acc = metrics.get("accuracy", 0.0)
        loss = metrics.get("loss", 0.0)
        entropy = metrics.get("entropy", 0.0)
        is_estimated = metrics.get("metrics_estimated", False)

        # Format technique name
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 28:
            display_name = display_name[:25] + "..."

        # Calculate improvement
        delta = (acc - baseline_acc) * 100

        # Format delta
        if tech_name == baseline_name:
            delta_str = "(baseline)"
        elif delta >= 0:
            delta_str = f"+{delta:.1f}%"
        else:
            delta_str = f"{delta:.1f}%"

        # Entropy suffix
        entropy_str = f"{entropy:.2f}"
        if is_estimated:
            entropy_str += "*"

        print(f"  {display_name:30} | {acc*100:>9.1f}% | {delta_str:>12} | {loss:>8.4f} | {entropy_str:>10}")


def _print_ascii_bar_chart(sorted_techniques):
    """
    Print ASCII bar chart for accuracy.

    Args:
        sorted_techniques: List of (technique_name, technique_data) sorted by accuracy
    """
    max_bar_width = 50  # Maximum width of bar in characters

    for tech_name, tech_data in sorted_techniques:
        metrics = tech_data.get("metrics", {})
        acc = metrics.get("accuracy", 0.0)

        # Format technique name
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."

        # Create bar
        bar_length = int(acc * max_bar_width)
        bar = "█" * bar_length

        # Print
        print(f"  {display_name:25} {bar:50} {acc*100:5.1f}%")


def _print_improvements(sorted_techniques, baseline_name, baseline_acc):
    """
    Print improvement over baseline with color coding.

    Args:
        sorted_techniques: List of (technique_name, technique_data) sorted by accuracy
        baseline_name: Name of baseline technique
        baseline_acc: Baseline accuracy value
    """
    # ANSI color codes for terminal
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

    for tech_name, tech_data in sorted_techniques:
        if tech_name == baseline_name:
            continue  # Skip baseline

        metrics = tech_data.get("metrics", {})
        acc = metrics.get("accuracy", 0.0)
        delta = (acc - baseline_acc) * 100

        # Format technique name
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        # Color code
        if delta > 0:
            color = GREEN
            sign = "+"
        elif delta < 0:
            color = RED
            sign = ""
        else:
            color = ""
            sign = ""

        if color:
            print(f"  {display_name:30} ... {color}{sign}{delta:.1f}%{RESET}")
        else:
            print(f"  {display_name:30} ... {sign}{delta:.1f}%")


def print_top_mistakes(results: Dict[str, Any], top_k: int = 5):
    """
    Print top mistakes made by the model.
    Enhanced with Rich table or ASCII table format.

    Args:
        results: Experimental results dictionary
        top_k: Number of top mistakes to show
    """
    print("\n❌ TOP MISTAKES:")
    print("-" * 80)

    # Collect all incorrect predictions
    mistakes = []
    techniques = results.get("techniques", {})

    for tech_name, tech_data in techniques.items():
        predictions = tech_data.get("predictions", [])
        for pred in predictions:
            if not pred.get("correct", False):
                mistakes.append({
                    "technique": tech_name,
                    "question": pred.get("question", "N/A"),
                    "expected": pred.get("ground_truth", "N/A"),
                    "predicted": pred.get("prediction", "N/A"),
                    "sample_id": pred.get("sample_id", "N/A"),
                })

    if not mistakes:
        print("  No mistakes found! 🎉")
        return

    # Use Rich table if available
    if RICH_AVAILABLE:
        _print_rich_mistakes_table(mistakes[:top_k])
    else:
        _print_ascii_mistakes_table(mistakes[:top_k])


def _print_rich_mistakes_table(mistakes):
    """
    Print mistakes using Rich table.

    Args:
        mistakes: List of mistake dictionaries
    """
    console = Console()
    table = Table(show_header=True, header_style="bold red")

    table.add_column("ID", style="cyan", width=10)
    table.add_column("Question", style="white", width=35)
    table.add_column("Expected", style="green", width=20)
    table.add_column("Got", style="red", width=20)
    table.add_column("Technique", style="yellow", width=15)

    for mistake in mistakes:
        # Truncate long texts
        question = mistake["question"]
        if len(question) > 60:
            question = question[:57] + "..."

        expected = mistake["expected"]
        if len(expected) > 35:
            expected = expected[:32] + "..."

        predicted = mistake["predicted"]
        if len(predicted) > 35:
            predicted = predicted[:32] + "..."

        technique = mistake["technique"].replace("_", " ").title()
        if len(technique) > 13:
            technique = technique[:10] + "..."

        table.add_row(
            str(mistake["sample_id"]),
            question,
            expected,
            predicted,
            technique
        )

    console.print(table)


def _print_ascii_mistakes_table(mistakes):
    """
    Print mistakes using ASCII table.

    Args:
        mistakes: List of mistake dictionaries
    """
    # Print header
    print(f"  {'ID':10} | {'Question':35} | {'Expected':20} | {'Got':20} | {'Technique':15}")
    print("  " + "-" * 110)

    # Print rows
    for mistake in mistakes:
        # Truncate long texts
        question = mistake["question"]
        if len(question) > 33:
            question = question[:30] + "..."

        expected = mistake["expected"]
        if len(expected) > 18:
            expected = expected[:15] + "..."

        predicted = mistake["predicted"]
        if len(predicted) > 18:
            predicted = predicted[:15] + "..."

        technique = mistake["technique"].replace("_", " ").title()
        if len(technique) > 13:
            technique = technique[:10] + "..."

        sample_id = str(mistake["sample_id"])
        if len(sample_id) > 10:
            sample_id = sample_id[:7] + "..."

        print(f"  {sample_id:10} | {question:35} | {expected:20} | {predicted:20} | {technique:15}")


def print_statistical_significance(stats: Dict[str, Any]):
    """
    Print statistical significance results.

    Args:
        stats: Statistical test results
    """
    print("\n📊 STATISTICAL SIGNIFICANCE:")
    print("-" * 80)

    alpha = stats.get("alpha", 0.05)
    bonferroni_alpha = stats.get("bonferroni_alpha", alpha)

    print(f"  Significance level (α): {alpha}")
    print(f"  Bonferroni-corrected α: {bonferroni_alpha:.4f}")

    pairwise = stats.get("pairwise_tests", {})
    significant_pairs = []

    for pair, result in pairwise.items():
        if result.get("significant", False):
            p_value = result.get("p_value", 1.0)
            significant_pairs.append((pair, p_value))

    if significant_pairs:
        print(f"\n  Significant differences found: {len(significant_pairs)}")
        for pair, p_value in sorted(significant_pairs, key=lambda x: x[1])[:5]:
            print(f"    • {pair}: p = {p_value:.4f} ✓")
    else:
        print("\n  No statistically significant differences detected.")


def generate_summary_dict(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary dictionary from full results.

    Args:
        results: Full experimental results

    Returns:
        Summary dictionary with key metrics
    """
    techniques = results.get("techniques", {})
    baseline_name = results.get("baseline_technique", "baseline")

    # Calculate baseline accuracy
    baseline_acc = 0.0
    if baseline_name in techniques:
        baseline_acc = techniques[baseline_name].get("metrics", {}).get("accuracy", 0.0)

    # Build summary
    summary = {
        "model": results.get("config", {}).get("llm_model", "unknown"),
        "baseline_accuracy": baseline_acc,
        "techniques": {},
        "best_technique": None,
        "max_improvement": 0.0,
    }

    # Process each technique
    max_improvement = 0.0
    best_technique = baseline_name

    for tech_name, tech_data in techniques.items():
        metrics = tech_data.get("metrics", {})
        acc = metrics.get("accuracy", 0.0)
        improvement = (acc - baseline_acc) * 100

        summary["techniques"][tech_name] = {
            "accuracy": acc,
            "loss": metrics.get("loss", 0.0),
            "entropy": metrics.get("entropy", 0.0),
            "perplexity": metrics.get("perplexity", 0.0),
            "improvement_over_baseline": improvement,
        }

        if improvement > max_improvement:
            max_improvement = improvement
            best_technique = tech_name

    summary["best_technique"] = best_technique
    summary["max_improvement"] = max_improvement

    return summary


def print_phase_header(phase_num: int, phase_name: str, description: str = ""):
    """
    Print a formatted phase header for CLI output.

    Args:
        phase_num: Phase number
        phase_name: Phase name
        description: Optional description
    """
    print(f"\n{'='*80}")
    print(f"[Phase {phase_num}] {phase_name}")
    if description:
        print(f"  → {description}")
    print(f"{'='*80}")


def print_progress(message: str, current: int = None, total: int = None):
    """
    Print progress message.

    Args:
        message: Progress message
        current: Current item number (optional)
        total: Total items (optional)
    """
    if current is not None and total is not None:
        percentage = (current / total * 100) if total > 0 else 0
        print(f"  [{current}/{total}] ({percentage:.0f}%) {message}")
    else:
        print(f"  • {message}")
