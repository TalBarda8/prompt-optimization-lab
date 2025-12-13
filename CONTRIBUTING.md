# Contributing to Prompt Optimization & Evaluation System

First off, thank you for considering contributing to this project! This document provides guidelines for contributing to the codebase.

---

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Development Setup](#development-setup)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Project Structure](#project-structure)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors.

### Our Standards

**Positive behavior includes:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints
- Gracefully accepting constructive criticism
- Focusing on what is best for the community

**Unacceptable behavior includes:**
- Harassment, trolling, or discriminatory comments
- Publishing others' private information
- Other conduct inappropriate in a professional setting

---

## How Can I Contribute?

### Reporting Bugs

**Before submitting a bug report:**
1. Check the [issue tracker](https://github.com/TalBarda8/prompt-optimization-lab/issues) for existing reports
2. Verify the bug with the latest version
3. Collect relevant information (OS, Python version, error messages)

**Bug report template:**

```markdown
**Environment:**
- OS: macOS 13.5
- Python: 3.10.8
- Version: 1.0.0

**Description:**
[Clear description of the bug]

**Steps to Reproduce:**
1. Run command: `python3 scripts/run_experiment.py ...`
2. Observe error: ...

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Error Logs:**
```
[Paste logs here]
```

**Additional Context:**
[Any other relevant information]
```

### Suggesting Enhancements

**Enhancement proposal template:**

```markdown
**Feature Request:**
[Clear title]

**Problem:**
[What problem does this solve?]

**Proposed Solution:**
[Describe your proposed solution]

**Alternatives Considered:**
[Other approaches you've considered]

**Additional Context:**
[Mockups, examples, references]
```

### Adding New Prompt Techniques

We welcome new prompt engineering techniques! Here's how:

**1. Create the technique:**

```python
# src/prompts/techniques.py

class MyNewTechnique(BasePromptBuilder):
    """
    My New Technique: Brief description.

    PRD Section: X.X
    Reference: [Paper or source]
    """

    def __init__(self):
        super().__init__(PromptTechnique.MY_TECHNIQUE)

    def build(self, fast_mode: bool = False, **kwargs) -> PromptTemplate:
        """Build the prompt template."""
        if fast_mode:
            system_prompt = "Brief instruction."
            user_prompt = "{question}\n\nAnswer directly."
        else:
            system_prompt = "Detailed instruction."
            user_prompt = "{question}\n\nDetailed reasoning structure."

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={"description": "My new technique"},
        )
```

**2. Add tests:**

```python
# tests/test_prompts.py

def test_my_new_technique():
    """Test MyNewTechnique prompt builder."""
    builder = MyNewTechnique()
    prompt = builder.build()

    assert prompt.technique == PromptTechnique.MY_TECHNIQUE
    assert prompt.system_prompt is not None
    assert "{question}" in prompt.user_prompt
```

**3. Update documentation:**
- Add to `docs/prompts/PROMPT_ENGINEERING_LOG.md`
- Include examples and rationale
- Document performance characteristics

### Adding New Metrics

**1. Implement the metric:**

```python
# src/metrics/custom_metric.py

def calculate_my_metric(
    predicted: str,
    ground_truth: str,
    **kwargs
) -> float:
    """
    Calculate my custom metric.

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Metric value (lower is better)
    """
    # Your implementation
    return metric_value
```

**2. Add tests:**

```python
# tests/test_metrics.py

def test_calculate_my_metric():
    """Test custom metric calculation."""
    score = calculate_my_metric(
        predicted="answer",
        ground_truth="answer"
    )
    assert 0 <= score <= 1
```

**3. Document in README and architecture docs**

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/prompt-optimization-lab.git
cd prompt-optimization-lab
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install package in editable mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"

# Or from requirements
pip install -r requirements.txt
```

### 4. Set Up Pre-commit Hooks (Optional)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install
```

### 5. Verify Setup

```bash
# Run tests
python3 -m pytest tests/

# Check coverage
python3 -m pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

**Line Length:**
- Maximum: 100 characters (not 79)
- Docstrings: 72 characters

**Imports:**
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# Local
from src.prompts import BasePromptBuilder
from src.metrics import calculate_entropy
```

**Naming Conventions:**
```python
# Classes: PascalCase
class PromptEvaluator:
    pass

# Functions: snake_case
def calculate_metrics():
    pass

# Constants: UPPER_CASE
MAX_RETRIES = 3

# Private: _leading_underscore
def _internal_function():
    pass
```

### Code Formatting

**Use Black for formatting:**

```bash
# Format all files
black src/ tests/

# Check without modifying
black --check src/
```

**Configuration (.pyproject.toml):**
```toml
[tool.black]
line-length = 100
target-version = ['py39']
```

### Type Hints

**Always use type hints:**

```python
from typing import List, Dict, Optional

def process_results(
    results: List[Dict[str, Any]],
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """Process experimental results."""
    # Implementation
    return metrics
```

### Docstrings

**Use Google style docstrings:**

```python
def calculate_loss(
    entropy: float,
    response_length: int,
    accuracy: float
) -> float:
    """
    Calculate composite loss function.

    Formula:
        L = 0.3·H + 0.2·|Y| + 0.3·(1-Acc)

    Args:
        entropy: Entropy in bits
        response_length: Response length in characters
        accuracy: Accuracy score (0-1)

    Returns:
        Composite loss value (lower is better)

    Example:
        >>> loss = calculate_loss(2.5, 100, 0.85)
        >>> print(f"Loss: {loss:.3f}")
        Loss: 0.875
    """
    # Implementation
    pass
```

---

## Testing Guidelines

### Test Coverage Requirements

**Minimum coverage:** 70%
**Target coverage:** 80%+

**Current coverage:** 66%

### Writing Tests

**1. Test file naming:**
```
tests/
├── test_data.py          # Tests for src/data/
├── test_llm.py           # Tests for src/llm/
├── test_prompts.py       # Tests for src/prompts/
├── test_metrics.py       # Tests for src/metrics/
└── test_pipeline.py      # Tests for src/pipeline/
```

**2. Test class naming:**
```python
class TestPromptEvaluator:
    """Tests for PromptEvaluator class."""

    def setup_method(self):
        """Setup before each test."""
        self.evaluator = PromptEvaluator()

    def test_initialization(self):
        """Test evaluator initialization."""
        assert self.evaluator is not None

    def test_evaluate_single_sample(self):
        """Test single sample evaluation."""
        result = self.evaluator.evaluate(sample)
        assert "accuracy" in result
```

**3. Use pytest fixtures:**
```python
import pytest

@pytest.fixture
def sample_data():
    """Provide sample test data."""
    return [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "Capital of France?", "answer": "Paris"},
    ]

def test_with_fixture(sample_data):
    """Test using fixture."""
    assert len(sample_data) == 2
```

### Running Tests

**Run all tests:**
```bash
python3 -m pytest tests/
```

**Run specific test file:**
```bash
python3 -m pytest tests/test_prompts.py
```

**Run specific test:**
```bash
python3 -m pytest tests/test_prompts.py::TestBaselinePrompt::test_build
```

**Run with coverage:**
```bash
python3 -m pytest --cov=src --cov-report=html
```

**Run fast (skip slow tests):**
```bash
python3 -m pytest -m "not slow"
```

### Test Markers

```python
import pytest

@pytest.mark.slow
def test_full_experiment():
    """Slow integration test."""
    pass

@pytest.mark.integration
def test_api_integration():
    """Integration test requiring API."""
    pass

@pytest.mark.unit
def test_calculate_entropy():
    """Fast unit test."""
    pass
```

---

## Documentation

### Documentation Requirements

**All code must be documented:**
1. Module-level docstrings
2. Class docstrings
3. Function/method docstrings
4. Complex logic comments

**Example:**

```python
"""
Prompt Engineering Module

This module implements various prompt engineering techniques for LLM optimization.
Techniques include CoT, ReAct, ToT, and more.

Author: Tal Barda
Date: 2025-12-13
"""

from typing import List, Optional


class PromptBuilder:
    """
    Base class for prompt builders.

    Provides common functionality for all prompt engineering techniques.
    Subclasses implement specific techniques by overriding `build()`.

    Attributes:
        technique: The prompt technique type
        metadata: Additional technique metadata
    """

    def __init__(self, technique: str):
        """
        Initialize prompt builder.

        Args:
            technique: Name of the prompt technique
        """
        self.technique = technique
        self.metadata = {}

    def build(self, **kwargs) -> str:
        """
        Build the prompt template.

        Must be implemented by subclasses.

        Args:
            **kwargs: Technique-specific parameters

        Returns:
            Formatted prompt string

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError("Subclasses must implement build()")
```

### Updating Documentation

**When adding features, update:**
1. README.md (if user-facing)
2. Architecture docs (if affecting design)
3. User manual (if changing usage)
4. Prompt log (if new technique)
5. Configuration guide (if new settings)

---

## Pull Request Process

### Before Submitting

**Checklist:**
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added (if applicable)
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No commented-out code
- [ ] No debug print statements

### Creating a Pull Request

**1. Create a feature branch:**
```bash
git checkout -b feature/my-new-feature
# or
git checkout -b fix/bug-description
```

**2. Make your changes and commit:**
```bash
git add .
git commit -m "Add my new feature

- Implemented X
- Updated Y
- Fixed Z

Closes #123"
```

**Commit message format:**
```
<type>: <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Formatting (no code change)
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance tasks

**3. Push and create PR:**
```bash
git push origin feature/my-new-feature
```

Then create PR on GitHub.

### PR Template

```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
[Describe testing performed]

## Checklist
- [ ] Tests pass
- [ ] Coverage ≥ 70%
- [ ] Documentation updated
- [ ] No linting errors
- [ ] Follows style guide

## Related Issues
Closes #123
```

### Code Review Process

**Reviewers will check:**
1. Code quality and style
2. Test coverage
3. Documentation completeness
4. Performance implications
5. Backward compatibility

**Response time:**
- Initial review: within 48 hours
- Follow-up: within 24 hours

---

## Project Structure

```
prompt-optimization-lab/
├── src/                      # Source code
│   ├── data/                 # Dataset loading
│   ├── llm/                  # LLM clients
│   ├── prompts/              # Prompt techniques
│   ├── metrics/              # Evaluation metrics
│   ├── visualization/        # Plots and reports
│   └── pipeline/             # Orchestration
├── tests/                    # Test suite
├── data/                     # Datasets
├── results/                  # Experiment outputs
├── docs/                     # Documentation
│   ├── architecture/         # Design docs
│   ├── guides/               # User guides
│   ├── prompts/              # Prompt engineering log
│   └── api/                  # API reference
├── scripts/                  # Utility scripts
├── config/                   # Configuration files
├── logs/                     # Log files
├── .github/                  # GitHub workflows
├── requirements.txt          # Dependencies
├── pyproject.toml            # Package configuration
├── README.md                 # Main documentation
├── CONTRIBUTING.md           # This file
└── LICENSE                   # MIT License
```

---

## Additional Resources

### Documentation
- [README](README.md) - Project overview
- [Architecture](docs/architecture/ARCHITECTURE.md) - System design
- [User Manual](docs/guides/USER_MANUAL.md) - Usage guide
- [Configuration](docs/guides/CONFIGURATION.md) - Configuration reference

### External Resources
- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Pytest Documentation](https://docs.pytest.org/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)

---

## Questions?

**Contact:**
- Email: tal.barda@example.com
- GitHub Issues: https://github.com/TalBarda8/prompt-optimization-lab/issues

---

Thank you for contributing! 🎉
