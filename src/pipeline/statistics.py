"""
Statistical Validation

Performs statistical significance testing (PRD Section 5.5).
"""

from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon


class StatisticalValidator:
    """
    Statistical significance testing for prompt comparisons.

    Implements:
    - Paired t-tests
    - Wilcoxon signed-rank tests
    - Bonferroni correction
    - Confidence intervals
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize validator.

        Args:
            alpha: Significance level (default: 0.05 for 95% confidence)
        """
        self.alpha = alpha

    def compare_techniques(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metric: str = "accuracy",
        use_parametric: bool = True,
    ) -> Dict[str, Any]:
        """
        Compare all techniques pairwise.

        Args:
            results: Dict mapping technique names to metric lists
            metric: Metric to compare (default: "accuracy")
            use_parametric: Use t-test (True) or Wilcoxon (False)

        Returns:
            Dictionary with p-values and significance decisions
        """
        techniques = list(results.keys())
        n = len(techniques)

        # Initialize results
        comparison = {
            "metric": metric,
            "test_type": "t-test" if use_parametric else "wilcoxon",
            "alpha": self.alpha,
            "bonferroni_alpha": self.alpha / (n * (n - 1) / 2) if n > 1 else self.alpha,
            "pairwise_tests": {},
            "p_value_matrix": pd.DataFrame(
                np.ones((n, n)),
                index=techniques,
                columns=techniques,
            ),
            "significance_matrix": pd.DataFrame(
                np.zeros((n, n), dtype=bool),
                index=techniques,
                columns=techniques,
            ),
        }

        # Perform pairwise comparisons
        for i, tech1 in enumerate(techniques):
            for j, tech2 in enumerate(techniques):
                if i >= j:
                    continue  # Skip diagonal and lower triangle

                data1 = results[tech1].get(metric, [])
                data2 = results[tech2].get(metric, [])

                if not data1 or not data2:
                    continue

                # Perform statistical test
                if use_parametric:
                    statistic, p_value = ttest_ind(data1, data2)
                else:
                    # Wilcoxon requires paired data
                    min_len = min(len(data1), len(data2))
                    statistic, p_value = wilcoxon(data1[:min_len], data2[:min_len])

                # Apply Bonferroni correction
                is_significant = p_value < comparison["bonferroni_alpha"]

                # Store results
                comparison["pairwise_tests"][f"{tech1}_vs_{tech2}"] = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "significant": is_significant,
                    "bonferroni_corrected": True,
                }

                # Update matrices
                comparison["p_value_matrix"].loc[tech1, tech2] = p_value
                comparison["p_value_matrix"].loc[tech2, tech1] = p_value
                comparison["significance_matrix"].loc[tech1, tech2] = is_significant
                comparison["significance_matrix"].loc[tech2, tech1] = is_significant

        return comparison

    def calculate_confidence_intervals(
        self,
        results: Dict[str, List[float]],
        confidence: float = 0.95,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for all techniques.

        Args:
            results: Dict mapping technique names to metric values
            confidence: Confidence level (default: 0.95)

        Returns:
            Dict with mean, lower, and upper bounds for each technique
        """
        intervals = {}

        for technique, values in results.items():
            if not values:
                continue

            mean = np.mean(values)
            std_err = stats.sem(values)
            ci = stats.t.interval(
                confidence,
                len(values) - 1,
                loc=mean,
                scale=std_err,
            )

            intervals[technique] = {
                "mean": float(mean),
                "lower": float(ci[0]),
                "upper": float(ci[1]),
                "std_error": float(std_err),
                "confidence": confidence,
            }

        return intervals

    def effect_size_cohen_d(
        self,
        group1: List[float],
        group2: List[float],
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1: First group values
            group2: Second group values

        Returns:
            Cohen's d value
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        # Cohen's d
        d = (mean1 - mean2) / pooled_std

        return float(d)

    def comprehensive_validation(
        self,
        results: Dict[str, Dict[str, List[float]]],
        metrics: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation.

        Args:
            results: Nested dict {technique: {metric: [values]}}
            metrics: List of metrics to validate (default: all)

        Returns:
            Comprehensive validation results
        """
        if metrics is None:
            # Infer metrics from first technique
            first_tech = next(iter(results.values()))
            metrics = list(first_tech.keys())

        validation = {
            "metrics_tested": metrics,
            "techniques": list(results.keys()),
            "alpha": self.alpha,
            "tests": {},
            "confidence_intervals": {},
            "effect_sizes": {},
        }

        # Test each metric
        for metric in metrics:
            # Extract metric data for all techniques
            metric_data = {
                tech: data.get(metric, [])
                for tech, data in results.items()
            }

            # Pairwise comparisons
            validation["tests"][metric] = self.compare_techniques(
                {tech: {metric: vals} for tech, vals in metric_data.items()},
                metric=metric,
                use_parametric=True,
            )

            # Confidence intervals
            validation["confidence_intervals"][metric] = (
                self.calculate_confidence_intervals(metric_data)
            )

            # Effect sizes (for first two techniques as example)
            techniques = list(results.keys())
            if len(techniques) >= 2:
                group1 = metric_data[techniques[0]]
                group2 = metric_data[techniques[1]]
                if group1 and group2:
                    validation["effect_sizes"][metric] = {
                        f"{techniques[0]}_vs_{techniques[1]}": self.effect_size_cohen_d(
                            group1, group2
                        )
                    }

        return validation
