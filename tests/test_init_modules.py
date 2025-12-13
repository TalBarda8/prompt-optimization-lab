"""
Unit tests for package __init__ modules

Tests that all package initializers can be imported and have correct exports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSrcInit:
    """Test src/__init__.py module."""

    def test_src_import(self):
        """Test importing src package."""
        import src
        assert hasattr(src, '__version__')
        assert hasattr(src, '__author__')
        assert src.__version__ == "1.0.0"
        assert src.__author__ == "Tal Barda"

    def test_src_exports(self):
        """Test src package exports."""
        import src
        assert 'data' in src.__all__
        assert 'llm' in src.__all__
        assert 'prompts' in src.__all__
        assert 'metrics' in src.__all__
        assert 'visualization' in src.__all__
        assert 'pipeline' in src.__all__

    def test_submodule_imports(self):
        """Test that submodules can be imported."""
        import src.data
        import src.llm
        import src.prompts
        import src.metrics
        import src.visualization
        import src.pipeline

        assert src.data is not None
        assert src.llm is not None
        assert src.prompts is not None
        assert src.metrics is not None
        assert src.visualization is not None
        assert src.pipeline is not None


class TestEvaluationInit:
    """Test src/evaluation/__init__.py module."""

    def test_evaluation_import(self):
        """Test importing evaluation package."""
        import src.evaluation
        assert src.evaluation is not None

    def test_evaluation_exports(self):
        """Test evaluation package exports."""
        import src.evaluation
        assert hasattr(src.evaluation, '__all__')
        assert src.evaluation.__all__ == []
