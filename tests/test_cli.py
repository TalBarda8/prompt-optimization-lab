"""
Unit tests for CLI

Tests command-line interface functionality.
"""

import sys
from pathlib import Path
import subprocess
import pytest

# Path to main.py
MAIN_SCRIPT = Path(__file__).parent.parent / "main.py"


class TestCLI:
    """Test CLI commands."""

    def test_help_command(self):
        """Test that help command works."""
        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Prompt Optimization Lab" in result.stdout
        assert "create-datasets" in result.stdout
        assert "run-experiment" in result.stdout

    def test_validate_command(self):
        """Test validate command."""
        # Check if datasets exist
        dataset_a = Path("data/dataset_a.json")
        dataset_b = Path("data/dataset_b.json")

        if not dataset_a.exists() or not dataset_b.exists():
            pytest.skip("Datasets not found")

        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT), "validate"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Validating" in result.stdout

    def test_run_experiment_help(self):
        """Test run-experiment help."""
        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT), "run-experiment", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--provider" in result.stdout
        assert "--techniques" in result.stdout

    def test_compare_help(self):
        """Test compare command help."""
        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT), "compare", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--techniques" in result.stdout

    def test_visualize_help(self):
        """Test visualize command help."""
        result = subprocess.run(
            [sys.executable, str(MAIN_SCRIPT), "visualize", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--results" in result.stdout


class TestHelperScripts:
    """Test helper scripts."""

    def test_analyze_results_help(self):
        """Test analyze_results.py help."""
        script = Path(__file__).parent.parent / "scripts" / "analyze_results.py"

        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Analyze experimental results" in result.stdout


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
