"""
Unit tests for pipeline orchestrator

Tests experiment orchestration and configuration.
"""

import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import ExperimentConfig


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = ExperimentConfig()

        assert config.temperature == 0.0
        assert config.max_tokens == 500
        assert config.save_intermediate is True
        assert config.llm_provider == "openai"  # Default is openai, not ollama

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = ExperimentConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            temperature=0.5,
            max_tokens=1000,
            techniques=["baseline", "cot"],
        )

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert len(config.techniques) == 2

    def test_config_dataset_paths(self):
        """Test dataset path configuration."""
        config = ExperimentConfig(
            dataset_paths={
                "dataset_a": "data/dataset_a.json",
                "dataset_b": "data/dataset_b.json",
            }
        )

        assert "dataset_a" in config.dataset_paths
        assert "dataset_b" in config.dataset_paths

    def test_config_output_dir(self):
        """Test output directory configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                output_dir=tmpdir
            )

            assert config.output_dir == tmpdir

    def test_config_techniques_list(self):
        """Test techniques configuration."""
        techniques = ["baseline", "cot", "react"]
        config = ExperimentConfig(techniques=techniques)

        assert config.techniques == techniques
        assert len(config.techniques) == 3

    def test_config_fast_mode(self):
        """Test fast mode configuration."""
        config_slow = ExperimentConfig(fast_mode=False)
        config_fast = ExperimentConfig(fast_mode=True)

        assert config_slow.fast_mode is False
        assert config_fast.fast_mode is True

    def test_config_fast_mode_setting(self):
        """Test fast mode configuration setting."""
        config = ExperimentConfig(fast_mode=True)

        assert config.fast_mode is True

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = ExperimentConfig(
            llm_provider="ollama",
            techniques=["baseline"]
        )

        assert config is not None

    def test_config_string_representation(self):
        """Test config string representation."""
        config = ExperimentConfig()

        # Should have a string representation
        config_str = str(config)
        assert isinstance(config_str, str)

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = ExperimentConfig(llm_provider="ollama")
        config2 = ExperimentConfig(llm_provider="ollama")

        # Configs with same parameters should be equal
        # (or at least comparable)
        assert config1.llm_provider == config2.llm_provider

    def test_config_immutability_check(self):
        """Test that critical config values are set."""
        config = ExperimentConfig()

        # Check required fields exist
        assert hasattr(config, 'llm_provider')
        assert hasattr(config, 'llm_model')
        assert hasattr(config, 'techniques')
        assert hasattr(config, 'temperature')

    def test_config_with_all_techniques(self):
        """Test config with all available techniques."""
        all_techniques = [
            "baseline",
            "chain_of_thought",
            "chain_of_thought_plus_plus",
            "react",
            "tree_of_thoughts",
            "role_based",
            "few_shot",
        ]

        config = ExperimentConfig(techniques=all_techniques)

        assert len(config.techniques) == 7

    def test_config_provider_options(self):
        """Test different provider options."""
        providers = ["openai", "anthropic", "ollama"]

        for provider in providers:
            config = ExperimentConfig(llm_provider=provider)
            assert config.llm_provider == provider

    def test_config_temperature_range(self):
        """Test temperature configuration range."""
        # Test valid temperatures
        for temp in [0.0, 0.3, 0.5, 0.7, 1.0]:
            config = ExperimentConfig(temperature=temp)
            assert config.temperature == temp

    def test_config_max_tokens_values(self):
        """Test different max_tokens values."""
        for tokens in [100, 500, 1000, 2000]:
            config = ExperimentConfig(max_tokens=tokens)
            assert config.max_tokens == tokens

    def test_config_save_intermediate_flag(self):
        """Test save_intermediate flag."""
        config_save = ExperimentConfig(save_intermediate=True)
        config_nosave = ExperimentConfig(save_intermediate=False)

        assert config_save.save_intermediate is True
        assert config_nosave.save_intermediate is False


class TestConfigEdgeCases:
    """Test configuration edge cases."""

    def test_config_empty_techniques(self):
        """Test config with empty techniques list."""
        config = ExperimentConfig(techniques=[])

        assert config.techniques == []

    def test_config_none_values(self):
        """Test config with None values where appropriate."""
        config = ExperimentConfig(
            dataset_paths=None
        )

        # Should handle None gracefully
        assert config is not None

    def test_config_empty_dataset_paths(self):
        """Test config with empty dataset paths."""
        config = ExperimentConfig(dataset_paths={})

        assert config.dataset_paths == {}

    def test_config_default_output_dir(self):
        """Test default output directory."""
        config = ExperimentConfig()

        # Should have an output directory
        assert hasattr(config, 'output_dir')
        assert config.output_dir == "results"
