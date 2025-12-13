"""
Unit tests for data.loaders module

Tests dataset loading and preprocessing functions.
"""

import sys
from pathlib import Path
import tempfile
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loaders import (
    load_dataset,
    save_dataset,
    DatasetLoader,
)


class TestDataLoaders:
    """Test dataset loading functions."""

    def setup_method(self):
        """Setup test data."""
        self.test_dataset = [
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "category": "arithmetic",
                "alternatives": ["four"],
            },
            {
                "question": "Capital of France?",
                "answer": "Paris",
                "category": "geography",
            },
        ]

        self.temp_dir = tempfile.mkdtemp()
        self.temp_file = Path(self.temp_dir) / "test_dataset.json"

        with open(self.temp_file, 'w') as f:
            json.dump(self.test_dataset, f)

    def teardown_method(self):
        """Cleanup."""
        import shutil
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_load_dataset(self):
        """Test loading a dataset from file."""
        dataset = load_dataset(str(self.temp_file))

        assert isinstance(dataset, list)
        assert len(dataset) == 2
        assert dataset[0]["question"] == "What is 2 + 2?"

    def test_load_dataset_nonexistent(self):
        """Test loading nonexistent dataset."""
        try:
            dataset = load_dataset("/nonexistent/path.json")
            # Should either raise exception or return empty list
            assert dataset is None or dataset == []
        except (FileNotFoundError, ValueError):
            # Expected behavior
            pass

    def test_save_dataset(self):
        """Test saving a dataset."""
        save_path = Path(self.temp_dir) / "saved_dataset.json"

        save_dataset({"data": self.test_dataset}, str(save_path))

        assert save_path.exists()

    def test_dataset_loader_class(self):
        """Test DatasetLoader class."""
        loader = DatasetLoader()

        assert loader is not None

    def test_load_dataset_invalid_json(self):
        """Test loading invalid JSON."""
        invalid_file = Path(self.temp_dir) / "invalid.json"
        invalid_file.write_text("{invalid json")

        try:
            dataset = load_dataset(str(invalid_file))
            # Should handle gracefully
            assert dataset is None or dataset == []
        except (json.JSONDecodeError, ValueError):
            # Expected behavior
            pass

    def test_dataset_roundtrip(self):
        """Test saving and loading a dataset."""
        save_path = Path(self.temp_dir) / "roundtrip.json"

        # Save
        dataset_dict = {"samples": self.test_dataset}
        save_dataset(dataset_dict, str(save_path))

        # Load
        loaded = load_dataset(str(save_path))

        assert loaded is not None
