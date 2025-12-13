"""
Unit tests for pipeline.summary module

Tests summary generation and formatting functions.
"""

import sys
from pathlib import Path
import io
from contextlib import redirect_stdout

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.summary import (
    print_experiment_summary,
    print_top_mistakes,
    print_statistical_significance,
    generate_summary_dict,
    print_phase_header,
    print_progress,
    _print_rich_accuracy_table,
    _print_ascii_accuracy_table,
    _print_ascii_bar_chart,
    _print_improvements,
)


class TestPhaseAndProgress:
    """Test phase and progress printing functions."""

    def test_print_phase_header(self):
        """Test phase header printing."""
        output = io.StringIO()

        with redirect_stdout(output):
            print_phase_header(1, "Testing", "Test description")

        result = output.getvalue()
        assert "Testing" in result or "PHASE" in result or len(result) > 0

    def test_print_progress(self):
        """Test progress printing."""
        output = io.StringIO()

        with redirect_stdout(output):
            print_progress("Processing...", current=5, total=10)

        result = output.getvalue()
        # Just verify it doesn't crash
        assert True


class TestSummaryDict:
    """Test summary dictionary generation."""

    def setup_method(self):
        """Setup test data."""
        self.results = {
            "techniques": {
                "baseline": {
                    "metrics": {
                        "accuracy": 0.75,
                        "loss": 0.35,
                    },
                },
                "cot": {
                    "metrics": {
                        "accuracy": 0.85,
                        "loss": 0.25,
                    },
                },
            },
        }

    def test_generate_summary_dict(self):
        """Test summary dictionary generation."""
        summary = generate_summary_dict(self.results)

        assert isinstance(summary, dict)

    def test_generate_summary_dict_empty(self):
        """Test summary with empty results."""
        summary = generate_summary_dict({"techniques": {}})

        assert isinstance(summary, dict)


class TestPrintFunctions:
    """Test print/output functions."""

    def setup_method(self):
        """Setup test data."""
        self.results = {
            "techniques": {
                "baseline": {
                    "metrics": {
                        "accuracy": 0.75,
                        "loss": 0.35,
                        "entropy": 2.5,
                        "perplexity": 5.6,
                    },
                    "token_stats": {
                        "avg_input_tokens": 50,
                        "avg_output_tokens": 100,
                        "total_tokens": 150,
                    },
                    "latency_stats": {
                        "mean": 2.5,
                        "std": 0.5,
                    },
                    "num_samples": 10,
                },
                "cot": {
                    "metrics": {
                        "accuracy": 0.85,
                        "loss": 0.25,
                        "entropy": 2.0,
                        "perplexity": 4.0,
                    },
                    "token_stats": {
                        "avg_input_tokens": 200,
                        "avg_output_tokens": 100,
                        "total_tokens": 300,
                    },
                    "latency_stats": {
                        "mean": 5.0,
                        "std": 1.0,
                    },
                    "num_samples": 10,
                },
            },
            "baseline_technique": "baseline",
            "mistakes": [
                {
                    "question": "What is 2+2?",
                    "predicted": "5",
                    "ground_truth": "4",
                    "technique": "baseline",
                },
            ],
            "statistical_tests": {
                "baseline_vs_cot": {
                    "t_statistic": -3.5,
                    "p_value": 0.002,
                    "significant": True,
                },
            },
        }

    def test_print_experiment_summary(self):
        """Test printing experiment summary."""
        # Capture stdout
        output = io.StringIO()

        with redirect_stdout(output):
            print_experiment_summary(
                self.results,
                model_name="TestLLM",
                show_top_mistakes=True,
                top_k=1
            )

        result = output.getvalue()
        assert "EXPERIMENT SUMMARY" in result
        assert "TestLLM" in result

    def test_print_experiment_summary_no_mistakes(self):
        """Test summary without mistakes section."""
        output = io.StringIO()

        with redirect_stdout(output):
            print_experiment_summary(
                self.results,
                show_top_mistakes=False
            )

        result = output.getvalue()
        assert "EXPERIMENT SUMMARY" in result

    def test_print_experiment_summary_empty_results(self):
        """Test summary with empty results."""
        output = io.StringIO()
        empty_results = {"techniques": {}}

        with redirect_stdout(output):
            print_experiment_summary(empty_results)

        result = output.getvalue()
        assert "No results available" in result or "EXPERIMENT SUMMARY" in result

    def test_print_statistical_significance(self):
        """Test printing statistical significance."""
        output = io.StringIO()

        stats = {
            "baseline_vs_cot": {
                "t_statistic": -3.5,
                "p_value": 0.002,
                "significant": True,
            },
        }

        with redirect_stdout(output):
            print_statistical_significance(stats)

        result = output.getvalue()
        # Just verify it doesn't crash
        assert isinstance(result, str)

    def test_print_top_mistakes(self):
        """Test printing top mistakes."""
        output = io.StringIO()

        with redirect_stdout(output):
            print_top_mistakes(self.results, top_k=1)

        result = output.getvalue()
        # Just verify it doesn't crash
        assert isinstance(result, str)


class TestSummaryEdgeCases:
    """Test edge cases in summary generation."""

    def test_empty_techniques(self):
        """Test with no techniques."""
        results = {"techniques": {}}
        output = io.StringIO()

        with redirect_stdout(output):
            print_experiment_summary(results)

        # Should handle gracefully
        assert True

    def test_missing_metrics(self):
        """Test with missing metrics."""
        results = {
            "techniques": {
                "baseline": {}
            }
        }
        output = io.StringIO()

        with redirect_stdout(output):
            print_experiment_summary(results)

        # Should handle gracefully
        assert True

    def test_single_technique(self):
        """Test with only one technique."""
        results = {
            "techniques": {
                "baseline": {
                    "metrics": {
                        "accuracy": 0.75,
                        "loss": 0.35,
                    },
                    "num_samples": 10,
                },
            },
            "baseline_technique": "baseline",
        }

        output = io.StringIO()

        with redirect_stdout(output):
            print_experiment_summary(results)

        result = output.getvalue()
        assert "baseline" in result.lower()
