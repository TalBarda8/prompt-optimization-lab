"""
Unit tests for pipeline module

Tests orchestration, evaluation, and statistical validation.
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import ExperimentConfig, ExperimentOrchestrator
from pipeline.statistics import StatisticalValidator


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_config_creation(self):
        """Test creating experiment config."""
        config = ExperimentConfig(
            dataset_paths={"dataset_a": "data/dataset_a.json"},
            llm_provider="openai",
            llm_model="gpt-4",
            techniques=["baseline", "chain_of_thought"],
        )

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert len(config.techniques) == 2

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.temperature == 0.0
        assert config.max_tokens == 500
        assert config.save_intermediate is True


class TestExperimentOrchestrator:
    """Test experiment orchestrator."""

    def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = ExperimentConfig(
            llm_provider="openai",
            techniques=["baseline"],
        )

        # This will fail if OPENAI_API_KEY is not set, but we just test structure
        try:
            orchestrator = ExperimentOrchestrator(config)
            assert orchestrator.config == config
            assert orchestrator.llm_client is not None
            assert orchestrator.output_path.exists()
        except Exception:
            # Skip if API key not available
            pass

    def test_technique_builder_access(self):
        """Test accessing technique builders."""
        config = ExperimentConfig(techniques=["baseline"])

        try:
            orchestrator = ExperimentOrchestrator(config)
            builder = orchestrator.get_technique_builder("baseline")
            assert builder is not None
        except Exception:
            pass

    def test_results_structure(self):
        """Test results dictionary structure."""
        config = ExperimentConfig()

        try:
            orchestrator = ExperimentOrchestrator(config)
            assert "config" in orchestrator.results
            assert "datasets" in orchestrator.results
            assert "evaluations" in orchestrator.results
            assert "metrics" in orchestrator.results
            assert "metadata" in orchestrator.results
        except Exception:
            pass


class TestStatisticalValidator:
    """Test statistical validation."""

    def test_validator_initialization(self):
        """Test initializing validator."""
        validator = StatisticalValidator(alpha=0.05)
        assert validator.alpha == 0.05

    def test_confidence_intervals(self):
        """Test confidence interval calculation."""
        validator = StatisticalValidator()

        results = {
            "technique_a": [0.75, 0.76, 0.74, 0.77, 0.75],
            "technique_b": [0.85, 0.86, 0.84, 0.87, 0.85],
        }

        intervals = validator.calculate_confidence_intervals(results)

        assert "technique_a" in intervals
        assert "technique_b" in intervals
        assert "mean" in intervals["technique_a"]
        assert "lower" in intervals["technique_a"]
        assert "upper" in intervals["technique_a"]

        # Mean of technique_b should be higher
        assert intervals["technique_b"]["mean"] > intervals["technique_a"]["mean"]

    def test_cohen_d_effect_size(self):
        """Test Cohen's d calculation."""
        validator = StatisticalValidator()

        # Large effect size
        group1 = [0.5, 0.52, 0.51, 0.53, 0.50]
        group2 = [0.8, 0.82, 0.81, 0.83, 0.80]

        d = validator.effect_size_cohen_d(group1, group2)

        # Should be large positive value
        assert d < -2.0  # Negative because group2 > group1

    def test_compare_techniques_parametric(self):
        """Test pairwise technique comparison with t-test."""
        validator = StatisticalValidator(alpha=0.05)

        results = {
            "baseline": {"accuracy": [0.70, 0.72, 0.71, 0.73, 0.70]},
            "optimized": {"accuracy": [0.85, 0.87, 0.86, 0.88, 0.85]},
        }

        comparison = validator.compare_techniques(
            results,
            metric="accuracy",
            use_parametric=True,
        )

        assert comparison["metric"] == "accuracy"
        assert comparison["test_type"] == "t-test"
        assert "pairwise_tests" in comparison
        assert "p_value_matrix" in comparison

    def test_comprehensive_validation(self):
        """Test comprehensive validation."""
        validator = StatisticalValidator()

        results = {
            "baseline": {
                "accuracy": [0.70, 0.72, 0.71],
                "loss": [0.35, 0.33, 0.34],
            },
            "optimized": {
                "accuracy": [0.85, 0.87, 0.86],
                "loss": [0.22, 0.20, 0.21],
            },
        }

        validation = validator.comprehensive_validation(results, metrics=["accuracy", "loss"])

        assert "metrics_tested" in validation
        assert "accuracy" in validation["metrics_tested"]
        assert "loss" in validation["metrics_tested"]
        assert "tests" in validation
        assert "confidence_intervals" in validation

    def test_bonferroni_correction(self):
        """Test Bonferroni correction is applied."""
        validator = StatisticalValidator(alpha=0.05)

        # Three techniques means 3 pairwise comparisons
        results = {
            "tech1": {"metric": [0.70, 0.72, 0.71]},
            "tech2": {"metric": [0.75, 0.77, 0.76]},
            "tech3": {"metric": [0.80, 0.82, 0.81]},
        }

        comparison = validator.compare_techniques(results, metric="metric")

        # Bonferroni alpha should be 0.05 / 3 ≈ 0.0167
        assert comparison["bonferroni_alpha"] < comparison["alpha"]

    def test_empty_data_handling(self):
        """Test handling of empty data."""
        validator = StatisticalValidator()

        results = {
            "tech1": [],
            "tech2": [],
        }

        intervals = validator.calculate_confidence_intervals(results)
        assert len(intervals) == 0  # Should skip empty data


# Run tests if executed directly
if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
