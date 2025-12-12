"""
Experiment Summary Generator

Generates human-readable summaries of experimental results for CLI output.
"""

from typing import Dict, List, Any, Optional
import numpy as np


def print_experiment_summary(
    results: Dict[str, Any],
    model_name: str = "LLM",
    show_top_mistakes: bool = True,
    top_k: int = 5,
):
    """
    Print comprehensive experiment summary to console.

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

    # Print accuracy table
    print("\n📊 ACCURACY BY TECHNIQUE:")
    print("-" * 80)

    # Sort techniques by accuracy
    sorted_techniques = sorted(
        techniques.items(),
        key=lambda x: x[1].get("metrics", {}).get("accuracy", 0.0),
        reverse=True
    )

    for tech_name, tech_data in sorted_techniques:
        metrics = tech_data.get("metrics", {})
        acc = metrics.get("accuracy", 0.0)
        improvement = (acc - baseline_acc) * 100

        # Format technique name
        display_name = tech_name.replace("_", " ").title()
        if len(display_name) > 30:
            display_name = display_name[:27] + "..."

        # Print row
        if tech_name == baseline_name:
            print(f"  {display_name:32} ... {acc*100:5.1f}%  (baseline)")
        elif improvement >= 0:
            print(f"  {display_name:32} ... {acc*100:5.1f}%  (+{improvement:4.1f}%)")
        else:
            print(f"  {display_name:32} ... {acc*100:5.1f}%  ({improvement:5.1f}%)")

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


def print_top_mistakes(results: Dict[str, Any], top_k: int = 5):
    """
    Print top mistakes made by the model.

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

    # Show top K mistakes
    for i, mistake in enumerate(mistakes[:top_k], 1):
        # Truncate long texts
        question = mistake["question"]
        if len(question) > 60:
            question = question[:57] + "..."

        expected = mistake["expected"]
        if len(expected) > 40:
            expected = expected[:37] + "..."

        predicted = mistake["predicted"]
        if len(predicted) > 40:
            predicted = predicted[:37] + "..."

        technique = mistake["technique"].replace("_", " ").title()

        print(f"\n  {i}. Sample: {mistake['sample_id']}")
        print(f"     Q: {question}")
        print(f"     Expected: {expected}")
        print(f"     Got:      {predicted}")
        print(f"     Technique: {technique}")


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
