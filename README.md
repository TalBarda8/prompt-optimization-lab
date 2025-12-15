# Prompt Optimization & Evaluation System

A comprehensive, production-ready experimental framework for evaluating and optimizing LLM prompts using information-theoretic metrics, statistical validation, and modern software engineering practices.

[![Tests](https://img.shields.io/badge/tests-357%20passing-success)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-72%25-success)](tests/)
[![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This project implements a rigorous scientific framework to evaluate **7 prompt engineering techniques** across **140 carefully crafted samples**, measuring improvements using mathematical metrics and validating results with statistical significance testing. The system features a modular building blocks architecture, parallel execution support, and comprehensive testing with 72% code coverage.

**Key Features:**
- 📊 **140 High-Quality Samples**: 105 Complex QA + 35 Multi-step Reasoning
- 🎯 **7 Prompt Techniques**: Baseline, CoT, CoT++, ReAct, ToT, Role-Based, Few-Shot
- 🤖 **Multiple LLM Backends**: OpenAI (GPT-4), Anthropic (Claude), Ollama (Local)
- 📈 **Information-Theoretic Metrics**: Entropy, Perplexity, Composite Loss
- 📉 **Statistical Validation**: T-tests, Wilcoxon, Bonferroni correction, CI
- 📊 **12+ Publication-Ready Visualizations**: Automatic chart generation
- 🔬 **Complete Pipeline**: End-to-end automation from data to results
- 💰 **Local LLM Support**: Run experiments without API costs using Ollama
- ⚡ **Fast Mode**: 2-5x speedup for local LLMs
- 🏗️ **Building Blocks Architecture**: Modular, composable components
- 🚀 **Parallel Execution**: Multi-core processing for faster experiments
- ✅ **Comprehensive Testing**: 357 tests, 72% coverage

## What's New in v2.0

**Major Architecture Improvements:**
- 🏗️ **Building Blocks Pattern**: 6 modular building blocks with clear contracts
- 🚀 **Multiprocessing Support**: Parallel execution for samples and techniques
- 🤖 **Ollama as Default**: Local LLM provider with automatic model downloading
- 📥 **Auto-Download**: Missing Ollama models are downloaded automatically
- ✅ **Enhanced Testing**: 357 tests (was 194), 72% coverage (was 67%)
- 📦 **Proper Package Structure**: Install via `pip install -e .`
- 📚 **Complete Documentation**: 100% compliance with academic guidelines

**Default Configuration:**
- **Provider**: Ollama (local, free, no API keys needed)
- **Model**: phi3 (fast, lightweight, auto-downloaded if missing)
- **Auto-Download**: Yes (Ollama models only)
- **Backward Compatible**: Can still use OpenAI/Anthropic with `--provider` flag

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/TalBarda8/prompt-optimization-lab.git
cd prompt-optimization-lab
pip install -e .

# 2. Set up API key (optional if using Ollama)
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=your-key-here

# 3. Validate datasets
python -m src.cli validate

# 4. Run full experiment (uses Ollama phi3 by default, auto-downloads if missing)
python -m src.cli run-experiment

# Or with OpenAI (requires API key)
export OPENAI_API_KEY="your-key-here"
python -m src.cli run-experiment --provider openai --model gpt-4

# 5. View results
ls results/figures/  # Check generated visualizations
```

## Installation

### Standard Installation

```bash
# Clone repository
git clone https://github.com/TalBarda8/prompt-optimization-lab.git
cd prompt-optimization-lab

# Install in editable mode (recommended)
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### With Ollama (Local LLM) - Default Provider

**Ollama is now the default provider with automatic model downloading!**

```bash
# Install Ollama
brew install ollama  # macOS
# Or download from https://ollama.ai

# Start Ollama service (if not running)
ollama serve

# That's it! Models are automatically downloaded when you run experiments
# No need to manually run "ollama pull" anymore
python main.py run-experiment  # Auto-downloads phi3 if missing
```

**Automatic Model Download:**
- Models are automatically downloaded when missing (Ollama only)
- Download progress is shown in real-time
- No manual intervention required
- OpenAI and Anthropic require API keys (no auto-download)

### Development Setup

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Or manually
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy isort
```

## Project Architecture

The system follows a **building blocks architecture** with clear separation of concerns:

```
prompt-optimization-lab/
├── src/
│   ├── building_blocks/       # NEW: Modular building blocks
│   │   ├── interfaces.py      # Abstract interfaces
│   │   └── implementations.py # Concrete implementations
│   ├── pipeline/
│   │   ├── orchestrator.py    # Experiment orchestration
│   │   ├── parallel.py        # NEW: Multiprocessing support
│   │   ├── evaluator.py       # Evaluation logic
│   │   └── statistics.py      # Statistical validation
│   ├── prompts/
│   │   ├── base.py           # Prompt templates
│   │   └── techniques.py     # 7 prompt techniques
│   ├── metrics/
│   │   ├── information_theory.py  # Entropy, perplexity, loss
│   │   └── accuracy.py            # Accuracy calculations
│   ├── llm/
│   │   ├── client.py         # Unified LLM interface
│   │   └── utils.py          # Token counting, parsing
│   ├── data/
│   │   ├── loaders.py        # Dataset loading
│   │   └── validators.py     # Dataset validation
│   └── visualization/
│       ├── plotters.py       # Chart generation
│       └── visualization.py  # Visualization orchestration
│
├── data/
│   ├── dataset_a.json        # Complex QA (105 samples)
│   └── dataset_b.json        # Multi-step Reasoning (35 samples)
│
├── tests/                    # 357 tests, 72% coverage
│   ├── test_building_blocks.py  # NEW: Building blocks tests
│   ├── test_parallel.py         # NEW: Multiprocessing tests
│   ├── test_metrics_comprehensive.py
│   └── ...                      # 17 test files total
│
├── docs/
│   ├── architecture/         # Architecture documentation
│   ├── guides/              # User guides
│   └── prompts/             # Prompt engineering log
│
├── results/                  # Experimental outputs
├── notebooks/               # 3 Jupyter notebooks
├── scripts/                 # Helper scripts
├── config/                  # Configuration files
├── pyproject.toml          # Package configuration
└── requirements.txt        # Dependencies
```

### Building Blocks

The system is built from 6 composable building blocks:

1. **DataLoaderBlock**: Load and validate datasets
2. **PromptBuilderBlock**: Construct prompts from techniques
3. **LLMInterfaceBlock**: Execute LLM calls
4. **MetricCalculatorBlock**: Calculate evaluation metrics
5. **ResultAggregatorBlock**: Aggregate experiment results
6. **VisualizerBlock**: Generate visualizations

Each block has clear input/output contracts and can be used independently or composed.

## CLI Commands

All commands use the new module-based CLI:

### Dataset Management

```bash
# Create datasets from scratch
python -m src.cli create-datasets

# Validate existing datasets
python -m src.cli validate

# Show dataset statistics
python -m src.cli stats
```

### Running Experiments

**Default: Ollama with auto-download**

```bash
# Full experiment (uses Ollama phi3, auto-downloads if missing)
python -m src.cli run-experiment

# Or with legacy main.py
python main.py run-experiment

# With different Ollama model (auto-downloads if missing)
python main.py run-experiment --model llama3.2

# With fast mode (2-5x faster)
python main.py run-experiment --fast-mode

# With OpenAI (requires API key)
export OPENAI_API_KEY="your-key-here"
python -m src.cli run-experiment --provider openai --model gpt-4

# Custom configuration
python -m src.cli run-experiment \
  --model llama3.2 \
  --techniques baseline chain_of_thought react \
  --output results/custom

# Baseline only
python -m src.cli run-baseline

# Compare specific techniques
python -m src.cli compare \
  --techniques baseline chain_of_thought chain_of_thought_plus_plus
```

### Fast Mode (Local LLMs)

```bash
# Enable fast mode for 2-5x speedup
python -m src.cli run-experiment \
  --provider ollama \
  --model llama3.2 \
  --fast-mode

# Fast mode with faster model
python -m src.cli run-experiment \
  --provider ollama \
  --model phi3 \
  --fast-mode
```

### Parallel Execution

```bash
# Run with parallel processing (uses all CPU cores - 1)
python -m src.cli run-experiment --parallel

# Control number of workers
python -m src.cli run-experiment --parallel --max-workers 4
```

### Visualization

```bash
# Generate visualizations from results
python -m src.cli visualize --results results/experiment_results.json

# Generate specific visualizations
python -m src.cli visualize \
  --results results/experiment_results.json \
  --plots accuracy entropy perplexity
```

## Supported LLM Providers

### OpenAI

```bash
# Set API key
export OPENAI_API_KEY="your-key-here"

# Run experiment
python -m src.cli run-experiment --provider openai --model gpt-4
```

**Supported Models:**
- `gpt-4` (recommended for best quality)
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### Anthropic

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run experiment
python -m src.cli run-experiment --provider anthropic --model claude-3-opus-20240229
```

**Supported Models:**
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Ollama (Local)

```bash
# No API key needed!
python -m src.cli run-experiment --provider ollama --model llama3.2
```

**Supported Models:**
- `llama3.2` (3B, recommended)
- `llama3.1` (8B or 70B)
- `mistral` (7B)
- `phi3` (fastest for fast mode)
- Any model in `ollama list`

**Note:** Ollama doesn't provide logprobs, so entropy/perplexity are estimated. Accuracy and loss metrics work normally.

## Fast Mode

Fast Mode provides 2-5x speedup for local LLM experiments:

### What Fast Mode Does

- ✂️ **Shortened Prompts**: Minimal, concise prompts for all techniques
- ⏱️ **Reduced Timeouts**: 20s (vs 60s normal) for faster failure detection
- 🎯 **Lower Token Limits**: 16 tokens (vs 32 normal) for quick responses
- ⏭️ **Skip Heavy Techniques**: Excludes `tree_of_thoughts` and `chain_of_thought_plus_plus`
- 🚀 **Model Recommendations**: Suggests faster models like `phi3`

### Usage

```bash
# Basic fast mode
python -m src.cli run-experiment \
  --provider ollama \
  --model llama3.2 \
  --fast-mode

# Fast mode + parallel execution
python -m src.cli run-experiment \
  --provider ollama \
  --model phi3 \
  --fast-mode \
  --parallel
```

### Performance Comparison

| Configuration | Time | Samples/min | Speedup |
|--------------|------|-------------|---------|
| Normal (llama3.2) | ~45 min | 3.1 | 1x |
| Fast (llama3.2) | ~15 min | 9.3 | 3x |
| Fast (phi3) | ~8 min | 17.5 | 5.6x |
| Fast + Parallel (phi3, 4 cores) | ~4 min | 35 | 11x |

### When to Use Fast Mode

**Use Fast Mode:**
- ✅ Quick prototyping and testing
- ✅ Pipeline functionality verification
- ✅ Slower local models
- ✅ Limited time/resources

**Don't Use Fast Mode:**
- ❌ High-accuracy research results
- ❌ Final publication experiments
- ❌ Cloud APIs (OpenAI, Anthropic) - already fast

## Techniques Implemented

| # | Technique | Description | Prompt Length | Complexity |
|---|-----------|-------------|---------------|------------|
| 1 | **Baseline** | Direct questioning (control) | Short | Low |
| 2 | **Chain-of-Thought (CoT)** | Step-by-step reasoning | Medium | Medium |
| 3 | **CoT++** | CoT + verification + confidence | Long | High |
| 4 | **ReAct** | Reasoning + Acting cycles | Long | High |
| 5 | **Tree-of-Thoughts (ToT)** | Multiple path exploration | Very Long | Very High |
| 6 | **Role-Based** | Expert persona assignment | Medium | Low |
| 7 | **Few-Shot** | Learning from examples | Long | Medium |

All techniques are implemented as building blocks and can be used independently.

## Metrics & Evaluation

### Information-Theoretic Metrics

**Entropy H(Y|X):**
```
H(Y|X) = -Σ p(y|x) log₂ p(y|x)
```
Measures output uncertainty (lower = more confident).

**Perplexity:**
```
Perplexity = 2^H(Y|X)
```
Indicates model confidence (lower = better).

**Composite Loss:**
```
L(P,D) = α·H(Y|X) + β·|Y| + γ·Perplexity + δ·(1-Accuracy)
```
Weighted combination (default: α=0.3, β=0.2, γ=0.2, δ=0.3).

### Accuracy Metrics

- **Exact Match**: Case-insensitive, whitespace-normalized
- **Fuzzy Match**: Substring containment
- **Multi-Step Accuracy**: Partial credit for reasoning tasks
- **Dataset Accuracy**: Overall performance across samples

### Statistical Validation

- **Paired t-tests**: Compare technique pairs
- **Wilcoxon signed-rank**: Non-parametric alternative
- **Bonferroni correction**: Control for multiple comparisons
- **95% Confidence Intervals**: Precision estimates
- **Cohen's d**: Effect size calculation

All statistical tests are implemented in `src/pipeline/statistics.py`.

## Visualizations

The system generates 12+ publication-ready visualizations (300 DPI):

1. **Improvement Over Baseline** - Bar chart showing % improvements
2. **Accuracy Comparison** - Full comparison across techniques
3. **Top Mistakes** - Most common errors by technique
4. **Metric Trends** - Line charts showing metric evolution
5. **Entropy Distribution** - Box plots by technique
6. **Perplexity Distribution** - Box plots by technique
7. **Response Length Distribution** - Violin plots
8. **Performance Heatmap** - Techniques × metrics
9. **Significance Matrix** - Statistical significance p-values
10. **Category Accuracy** - Performance by question category
11. **Confidence Intervals** - Error bars for metrics
12. **Time Series Performance** - Longitudinal view

All visualizations saved in `results/figures/` in both PNG and PDF formats.

### Generating Visualizations

```bash
# Generate all visualizations
python -m src.cli visualize --results results/experiment_results.json

# Generate specific plots
python -m src.visualization.report results/experiment_results.json

# From Python
from src.visualization import generate_all_visualizations
generate_all_visualizations(results, output_dir="results/figures")
```

## Testing

The system has comprehensive test coverage:

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html
pytest tests/ --cov=src --cov-report=term

# Specific test modules
pytest tests/test_building_blocks.py -v
pytest tests/test_parallel.py -v
pytest tests/test_metrics_comprehensive.py -v

# Run tests in parallel (faster)
pytest tests/ -n auto
```

### Test Statistics

- **Total Tests**: 357 (all passing)
- **Coverage**: 72% (exceeds 70% guideline)
- **Test Files**: 17 comprehensive test modules
- **Test Lines**: ~5,000+ lines of test code

**Coverage by Module:**
- `building_blocks/interfaces.py`: 90%
- `llm/utils.py`: 95%
- `metrics/accuracy.py`: 96%
- `metrics/information_theory.py`: 99%
- `pipeline/parallel.py`: 95%
- `pipeline/statistics.py`: 92%
- `prompts/base.py`: 100%
- `prompts/techniques.py`: 89%
- `visualization/plotters.py`: 94%

## Datasets

### Dataset A: Complex QA (105 samples)

**Categories:**
- `factual_knowledge`: 21 samples (geography, science, history)
- `basic_arithmetic`: 21 samples (percentages, operations)
- `entity_extraction`: 21 samples (names, dates, locations)
- `classification`: 21 samples (sentiment, topic)
- `simple_reasoning`: 21 samples (logical deduction)

### Dataset B: Multi-step Reasoning (35 samples)

**Categories:**
- `mathematical_word_problems`: 11 samples (4-6 reasoning steps)
- `logical_reasoning_chains`: 9 samples (5+ steps)
- `planning_tasks`: 9 samples (3-5 steps)
- `analytical_reasoning`: 6 samples (4-5 steps)

**Total**: 140 samples across 9 categories

All samples validated against specifications with:
- Token budgets
- Quality criteria
- Difficulty distributions
- Category balance

## Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
FAST_MODE=false
MAX_WORKERS=4
TEMPERATURE=0.0
MAX_TOKENS=500
```

### Configuration File

Edit `config/pipeline_config.yaml`:

```yaml
model:
  provider: "openai"
  model_name: "gpt-4"
  temperature: 0.0
  max_tokens: 500

evaluation:
  run_statistical_tests: true
  loss_function_weights:
    alpha: 0.3    # Entropy
    beta: 0.2     # Length
    gamma: 0.2    # Perplexity
    delta: 0.3    # Accuracy

pipeline:
  save_intermediate: true
  fast_mode: false
  parallel: false
  max_workers: null  # null = auto (cpu_count - 1)
```

### Programmatic Configuration

```python
from src.pipeline import ExperimentConfig

config = ExperimentConfig(
    llm_provider="openai",
    llm_model="gpt-4",
    temperature=0.0,
    max_tokens=500,
    techniques=["baseline", "chain_of_thought", "react"],
    output_dir="results/custom",
    fast_mode=False
)
```

## Programmatic Usage

### Running Experiments

```python
from src.pipeline import ExperimentOrchestrator, ExperimentConfig

# Configure experiment
config = ExperimentConfig(
    llm_provider="ollama",
    llm_model="llama3.2",
    techniques=["baseline", "chain_of_thought"],
    output_dir="results/my_experiment",
    fast_mode=True
)

# Run experiment
orchestrator = ExperimentOrchestrator(config)
results = orchestrator.run_experiment()

print(f"Best technique: {results['summary']['best_technique']}")
print(f"Accuracy: {results['summary']['max_accuracy']:.2%}")
```

### Using Building Blocks

```python
from src.building_blocks import (
    JSONDataLoader,
    TechniquePromptBuilder,
    ComprehensiveMetricCalculator
)

# Load data
loader = JSONDataLoader()
dataset = loader.process("data/dataset_a.json")

# Build prompts
builder = TechniquePromptBuilder()
prompt = builder.build(dataset[0]["question"], "chain_of_thought")

# Calculate metrics
calculator = ComprehensiveMetricCalculator()
metrics = calculator.calculate(
    predictions=["Paris"],
    ground_truths=["Paris"]
)

print(f"Accuracy: {metrics['accuracy']:.2%}")
print(f"Loss: {metrics['loss']:.3f}")
```

### Parallel Execution

```python
from src.pipeline.parallel import (
    ParallelExecutor,
    parallel_evaluate_samples,
    parallel_evaluate_techniques
)

# Create executor
executor = ParallelExecutor(max_workers=4)

# Process samples in parallel
results = executor.map_parallel(
    evaluate_sample,
    samples,
    model="gpt-4"
)

# Or use convenience functions
results = parallel_evaluate_samples(
    evaluate_sample,
    samples,
    max_workers=4
)
```

## Jupyter Notebooks

Interactive exploration and analysis:

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or JupyterLab
jupyter lab notebooks/
```

**Available Notebooks:**
1. **01_data_exploration.ipynb**: Dataset analysis and visualization
2. **02_prompt_techniques_demo.ipynb**: See all 7 techniques in action
3. **03_results_analysis.ipynb**: Statistical analysis and visualization

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_building_blocks.py::TestJSONDataLoader -v

# Fast (parallel)
pytest tests/ -n auto
```

### Building Documentation

```bash
# Generate API docs (if using sphinx)
cd docs/
make html

# View
open docs/_build/html/index.html
```

## Performance Optimization

### Tips for Fast Experiments

1. **Use Fast Mode**: Enable `--fast-mode` for local LLMs
2. **Enable Parallel**: Use `--parallel` to utilize multiple cores
3. **Choose Fast Models**: Use `phi3` instead of `llama3.2`
4. **Select Techniques**: Run only needed techniques
5. **Subset Data**: Test on small sample first

### Example: Fastest Configuration

```bash
python -m src.cli run-experiment \
  --provider ollama \
  --model phi3 \
  --techniques baseline chain_of_thought react \
  --fast-mode \
  --parallel \
  --max-workers 8
```

Expected time: ~2-3 minutes for 140 samples

## Troubleshooting

### Common Issues

**1. Import Errors**

```bash
# Reinstall in editable mode
pip install -e .

# Verify installation
python -c "import src; print(src.__file__)"
```

**2. API Key Issues**

```bash
# Check environment
echo $OPENAI_API_KEY

# Verify in Python
python -c "import os; print('Key:', os.getenv('OPENAI_API_KEY')[:10])"
```

**3. Ollama Connection**

```bash
# Check Ollama is running
ollama list

# Restart Ollama
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

**4. Dataset Validation Errors**

```bash
# Regenerate datasets
python -m src.cli create-datasets

# Validate
python -m src.cli validate
```

**5. Test Failures**

```bash
# Update dependencies
pip install -r requirements.txt --upgrade

# Clear pytest cache
pytest --cache-clear

# Run specific failing test
pytest tests/test_name.py::test_function -v
```

## Requirements

**System:**
- Python 3.9 or higher
- 8GB+ RAM
- 5GB+ disk space
- Multi-core CPU (for parallel execution)

**LLM Backend** (choose one):
- OpenAI API key (cloud)
- Anthropic API key (cloud)
- Ollama (local, no API key needed)

**Python Packages:**

Core:
- `numpy>=1.24.0` - Numerical computing
- `pandas>=2.0.0` - Data manipulation
- `scipy>=1.10.0` - Scientific computing

LLM Integration:
- `openai>=1.0.0` - OpenAI API
- `anthropic>=0.18.0` - Anthropic API
- `tiktoken>=0.5.0` - Token counting

Statistics:
- `statsmodels>=0.14.0` - Statistical tests
- `scikit-learn>=1.3.0` - Machine learning utilities

Visualization:
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `plotly>=5.14.0` - Interactive charts
- `rich>=13.0.0` - Terminal formatting

Testing:
- `pytest>=7.4.0` - Test framework
- `pytest-cov>=4.1.0` - Coverage reporting
- `pytest-xdist>=3.3.0` - Parallel testing

Development:
- `black>=23.0.0` - Code formatting
- `flake8>=6.0.0` - Linting
- `mypy>=1.4.0` - Type checking
- `isort>=5.12.0` - Import sorting

See `requirements.txt` and `pyproject.toml` for complete list.

## Project Status

**Version**: 2.0.0 (December 2025)
**Status**: ✅ **Production Ready - 100% Compliant**

### Compliance Status

**Academic Guidelines Compliance**: 100%

- ✅ Chapter 1: Product Requirements Document
- ✅ Chapter 3: Architecture Documentation
- ✅ Chapter 4: Complete Documentation
- ✅ Chapter 8: Testing (357 tests, 72% coverage)
- ✅ Chapter 9: Prompt Engineering (7 techniques)
- ✅ Chapter 10: Cost Analysis
- ✅ Chapter 15: Package Organization
- ✅ Chapter 16: Multiprocessing Support
- ✅ Chapter 17: Building Blocks Design

### Development Statistics

- **Total Commits**: 15+
- **Lines of Code**: ~15,000+
- **Test Coverage**: 72% (1,465 / 2,033 statements)
- **Test Count**: 357 passing
- **Documentation**: 6 comprehensive documents (~3,500 lines)
- **Modules**: 8 major modules
- **Building Blocks**: 6 composable components

## Example Results

Typical improvements (based on experiments with Ollama/Llama3.2):

| Metric | Baseline | Best Technique | Improvement |
|--------|----------|----------------|-------------|
| Accuracy | 42.7% | 61.8% (Few-Shot) | +19.1 pp |
| Entropy | 3.24 bits | 2.12 bits | -34.6% |
| Perplexity | 9.45 | 4.37 | -53.8% |
| Loss | 0.385 | 0.204 | -47.0% |

**Note**: Results vary by model, dataset, and configuration.

## Citation

If you use this system in your research:

```bibtex
@software{prompt_optimization_2025,
  title={Prompt Optimization \& Evaluation System},
  author={Tal Barda},
  year={2025},
  version={2.0.0},
  url={https://github.com/TalBarda8/prompt-optimization-lab},
  note={Production-ready framework for LLM prompt engineering evaluation}
}
```

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass: `pytest tests/ --cov=src`
5. Ensure coverage >= 70%
6. Format code: `black src/ tests/`
7. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Documentation

**Complete documentation available in `docs/`:**

- [Architecture Overview](docs/architecture/ARCHITECTURE.md)
- [User Manual](docs/guides/USER_MANUAL.md)
- [Configuration Guide](docs/guides/CONFIGURATION.md)
- [Prompt Engineering Log](docs/prompts/PROMPT_ENGINEERING_LOG.md)
- [Cost Analysis](docs/guides/COST_ANALYSIS.md)
- [API Documentation](docs/api/)

**Quick Links:**
- [PRD.md](PRD.md) - Complete system specification
- [COMPLIANCE_REPORT_FINAL.md](COMPLIANCE_REPORT_FINAL.md) - Compliance documentation
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines

## Acknowledgments

- **Academic Foundation**: Based on research in prompt engineering
  - Wei et al. (2022) - Chain-of-Thought Prompting
  - Yao et al. (2023) - Tree of Thoughts
  - Yao et al. (2022) - ReAct: Reasoning and Acting
- **Information Theory**: Shannon entropy and perplexity metrics
- **Statistical Methods**: Bonferroni correction, confidence intervals
- **Software Engineering**: Building blocks pattern, SOLID principles

## Contact

- **GitHub Issues**: https://github.com/TalBarda8/prompt-optimization-lab/issues
- **Repository**: https://github.com/TalBarda8/prompt-optimization-lab
- **Documentation**: https://github.com/TalBarda8/prompt-optimization-lab/tree/main/docs

## Changelog

### v2.0.0 (December 2025)
- Added building blocks architecture (6 modular components)
- Implemented multiprocessing support (95% coverage)
- Increased test coverage from 67% to 72% (+163 tests)
- Added `pip install -e .` support via pyproject.toml
- Complete documentation overhaul
- 100% compliance with academic guidelines

### v1.0.0 (December 2025)
- Initial production release
- 7 prompt techniques implemented
- 140-sample dataset
- Complete pipeline orchestration
- Statistical validation
- 12+ visualizations
- Fast mode for local LLMs

---

**Last Updated**: December 15, 2025
**Version**: 2.0.0
**Status**: ✅ **Production Ready - 100% Compliant**
