"""
Unit tests for visualization module

Tests all 12 visualization functions.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization import (
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


class TestVisualizationPlots:
    """Test all visualization plotting functions."""

    def setup_method(self):
        """Setup test data before each test."""
        self.techniques = ["Baseline", "CoT", "CoT++", "ReAct"]

        self.accuracy_data = {
            "Baseline": 0.75,
            "CoT": 0.85,
            "CoT++": 0.88,
            "ReAct": 0.82,
        }

        self.loss_data = {
            "Baseline": 0.35,
            "CoT": 0.25,
            "CoT++": 0.22,
            "ReAct": 0.28,
        }

        self.entropy_dist = {
            "Baseline": [2.5, 2.7, 2.6, 2.8],
            "CoT": [2.0, 2.1, 2.2, 2.0],
            "CoT++": [1.8, 1.9, 1.7, 1.9],
            "ReAct": [2.3, 2.4, 2.2, 2.3],
        }

        self.perplexity_dist = {
            "Baseline": [5.6, 6.5, 6.2, 7.0],
            "CoT": [4.0, 4.2, 4.5, 4.0],
            "CoT++": [3.5, 3.7, 3.3, 3.7],
            "ReAct": [4.9, 5.1, 4.7, 4.9],
        }

        self.length_dist = {
            "Baseline": [50, 55, 52, 58],
            "CoT": [120, 125, 118, 130],
            "CoT++": [150, 155, 148, 160],
            "ReAct": [180, 185, 178, 190],
        }

    def teardown_method(self):
        """Cleanup after each test."""
        plt.close('all')

    def test_plot_accuracy_comparison(self):
        """Test accuracy comparison plot."""
        fig = plot_accuracy_comparison(self.accuracy_data)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_accuracy_comparison_save(self):
        """Test saving accuracy plot."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            fig = plot_accuracy_comparison(self.accuracy_data, save_path=tmp.name)
            assert Path(tmp.name).exists()
            plt.close(fig)
            Path(tmp.name).unlink()

    def test_plot_loss_comparison(self):
        """Test loss comparison plot."""
        fig = plot_loss_comparison(self.loss_data)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_entropy_distribution(self):
        """Test entropy distribution plot."""
        fig = plot_entropy_distribution(self.entropy_dist)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_perplexity_distribution(self):
        """Test perplexity distribution plot."""
        fig = plot_perplexity_distribution(self.perplexity_dist)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_response_length_distribution(self):
        """Test response length distribution plot."""
        fig = plot_response_length_distribution(self.length_dist)
        assert fig is not None
        assert len(fig.axes) == 1
        plt.close(fig)

    def test_plot_performance_heatmap(self):
        """Test performance heatmap."""
        data = pd.DataFrame({
            'Accuracy': [0.75, 0.85, 0.88, 0.82],
            'Entropy': [2.6, 2.1, 1.8, 2.3],
            'Perplexity': [6.3, 4.2, 3.6, 4.9],
        }, index=self.techniques)

        fig = plot_performance_heatmap(data)
        assert fig is not None
        assert len(fig.axes) == 2  # Heatmap + colorbar
        plt.close(fig)

    def test_plot_significance_matrix(self):
        """Test significance matrix plot."""
        p_values = pd.DataFrame(
            [[1.0, 0.05, 0.01, 0.03],
             [0.05, 1.0, 0.02, 0.04],
             [0.01, 0.02, 1.0, 0.06],
             [0.03, 0.04, 0.06, 1.0]],
            index=self.techniques,
            columns=self.techniques,
        )

        fig = plot_significance_matrix(p_values)
        assert fig is not None
        plt.close(fig)

    def test_plot_category_accuracy(self):
        """Test category accuracy plot."""
        data = {
            "Baseline": {"math": 0.70, "logic": 0.75, "factual": 0.80},
            "CoT": {"math": 0.85, "logic": 0.82, "factual": 0.88},
            "CoT++": {"math": 0.90, "logic": 0.85, "factual": 0.90},
        }

        fig = plot_category_accuracy(data)
        assert fig is not None
        plt.close(fig)

    def test_plot_confidence_intervals(self):
        """Test confidence intervals plot."""
        means = {"Baseline": 0.75, "CoT": 0.85, "CoT++": 0.88}
        ci_lower = {"Baseline": 0.70, "CoT": 0.82, "CoT++": 0.85}
        ci_upper = {"Baseline": 0.80, "CoT": 0.88, "CoT++": 0.91}

        fig = plot_confidence_intervals(means, ci_lower, ci_upper)
        assert fig is not None
        plt.close(fig)

    def test_plot_time_series_performance(self):
        """Test time series performance plot."""
        data = {
            "Baseline": [(0, 0.5), (1, 0.6), (2, 0.65), (3, 0.70)],
            "CoT": [(0, 0.7), (1, 0.75), (2, 0.80), (3, 0.85)],
        }

        fig = plot_time_series_performance(data)
        assert fig is not None
        plt.close(fig)

    def test_plot_correlation_matrix(self):
        """Test correlation matrix plot."""
        data = pd.DataFrame({
            'Accuracy': [0.75, 0.85, 0.88, 0.82],
            'Entropy': [2.6, 2.1, 1.8, 2.3],
            'Perplexity': [6.3, 4.2, 3.6, 4.9],
            'Loss': [0.35, 0.25, 0.22, 0.28],
        })

        fig = plot_correlation_matrix(data)
        assert fig is not None
        plt.close(fig)

    def test_plot_technique_rankings(self):
        """Test technique rankings plot."""
        rankings = {"Baseline": 4, "CoT": 2, "CoT++": 1, "ReAct": 3}
        scores = {"Baseline": 0.75, "CoT": 0.85, "CoT++": 0.88, "ReAct": 0.82}

        fig = plot_technique_rankings(rankings, scores)
        assert fig is not None
        plt.close(fig)


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""

    def teardown_method(self):
        """Cleanup after each test."""
        plt.close('all')

    def test_empty_data_accuracy(self):
        """Test with empty accuracy data."""
        fig = plot_accuracy_comparison({})
        assert fig is not None
        plt.close(fig)

    def test_single_technique(self):
        """Test with single technique."""
        data = {"Baseline": 0.75}
        fig = plot_accuracy_comparison(data)
        assert fig is not None
        plt.close(fig)

    def test_large_number_of_techniques(self):
        """Test with many techniques."""
        data = {f"Technique_{i}": 0.5 + i*0.01 for i in range(20)}
        fig = plot_accuracy_comparison(data)
        assert fig is not None
        plt.close(fig)


# Run tests if executed directly
if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
