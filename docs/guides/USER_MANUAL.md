# User Manual: Prompt Optimization & Evaluation System

**Version:** 2.0.0
**Author:** Tal Barda
**Last Updated:** December 15, 2025

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Quick Start](#3-quick-start)
4. [Configuration](#4-configuration)
5. [Running Experiments](#5-running-experiments)
6. [Understanding Results](#6-understanding-results)
7. [Advanced Usage](#7-advanced-usage)
8. [Troubleshooting](#8-troubleshooting)
9. [FAQ](#9-faq)
10. [Support](#10-support)

---

## 1. Introduction

### 1.1 What is This System?

The Prompt Optimization & Evaluation System is a comprehensive experimental framework designed to evaluate and optimize Large Language Model (LLM) prompts using rigorous information-theoretic metrics and statistical validation.

### 1.2 Key Features

- **7 Prompt Engineering Techniques**: Baseline, CoT, CoT++, ReAct, ToT, Role-Based, Few-Shot
- **Advanced Metrics**: Entropy, Perplexity, Composite Loss, Accuracy
- **Statistical Validation**: t-tests, Wilcoxon, Bonferroni correction
- **Multiple LLM Backends**: OpenAI, Anthropic, Ollama (local)
- **Comprehensive Visualization**: 12+ publication-ready charts
- **Fast Mode**: 4× speedup for production use
- **Multiprocessing Support**: Parallel execution for performance (NEW in v2.0)
- **Building Blocks Architecture**: 6 modular composable components (NEW in v2.0)
- **Comprehensive Testing**: 357 tests with 72% coverage (NEW in v2.0)
- **Reproducible**: Fixed random seeds, deterministic evaluation

### 1.3 Who Should Use This?

- **Researchers** studying prompt engineering effectiveness
- **ML Engineers** optimizing LLM applications
- **Data Scientists** analyzing model performance
- **Students** learning about prompt optimization

---

## 2. Installation

### 2.1 System Requirements

**Minimum:**
- Python 3.9+
- 8 GB RAM
- 5 GB disk space

**Recommended:**
- Python 3.10+
- 16 GB RAM
- GPU (optional, for faster local models)

### 2.2 Installation Steps

**Step 1: Clone the Repository**

```bash
git clone https://github.com/TalBarda8/prompt-optimization-lab.git
cd prompt-optimization-lab
```

**Step 2: Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**Step 4: Install Package (Recommended)**

```bash
pip install -e .
```

This enables you to use `python -m src.cli` commands from anywhere.

### 2.3 Verify Installation

```bash
python3 -m pytest tests/ -v
```

Expected output: All tests should pass (**357 tests**, 72% coverage).

---

## 3. Quick Start

### 3.1 Your First Experiment (5 Minutes)

**Step 1: Set Up API Keys (if using cloud LLMs)**

```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

**Step 2: Run Example Experiment**

```bash
python -m src.cli run-experiment \
    --provider ollama \
    --model llama3.2 \
    --techniques baseline cot react \
    --output results/
```

Or use the legacy script:

```bash
python3 scripts/run_experiment.py \
    --dataset data/dataset_a.json \
    --llm-provider ollama \
    --techniques baseline cot react \
    --output results/
```

**Step 3: View Results**

```bash
cat results/experiment_report.md
```

### 3.2 Expected Output

```
EXPERIMENT SUMMARY (Llama 3.2)
================================================================================

Technique Rankings (by Accuracy):
1. 🥇 ReAct:    85.2% (+10.2% vs baseline)
2. 🥈 CoT:      82.1% (+7.1% vs baseline)
3.    Baseline: 75.0%

✓ Statistical Significance: ReAct vs Baseline (p=0.003)
```

---

## 4. Configuration

### 4.1 Configuration File

Create `config.yaml`:

```yaml
# LLM Configuration
llm:
  provider: ollama  # Options: openai, anthropic, ollama
  model: llama3.2  # Model name
  temperature: 0.0  # 0.0 = deterministic
  max_tokens: 500

# Experiment Configuration
experiment:
  techniques:
    - baseline
    - chain_of_thought
    - react
  fast_mode: false  # Set true for 4× speedup
  random_seed: 42

# Dataset Configuration
datasets:
  dataset_a: data/dataset_a.json
  dataset_b: data/dataset_b.json

# Output Configuration
output:
  directory: results/
  save_intermediate: true
  generate_visualizations: true
```

### 4.2 Environment Variables

```bash
# Required (for cloud LLMs)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional
export FAST_MODE=true
export MAX_SAMPLES=100
export RANDOM_SEED=42
```

### 4.3 Command-Line Arguments

```bash
python3 scripts/run_experiment.py \
    --config config.yaml \
    --dataset data/dataset_a.json \
    --llm-provider ollama \
    --llm-model llama3.2 \
    --techniques baseline cot react \
    --temperature 0.0 \
    --max-tokens 500 \
    --fast-mode \
    --output results/ \
    --visualize
```

---

## 5. Running Experiments

### 5.1 Basic Experiment

**Run single technique:**

```bash
python3 scripts/run_experiment.py \
    --dataset data/dataset_a.json \
    --techniques chain_of_thought \
    --llm-provider ollama
```

**Run multiple techniques:**

```bash
python3 scripts/run_experiment.py \
    --dataset data/dataset_a.json \
    --techniques baseline cot cot_plus_plus react tot role_based few_shot \
    --llm-provider ollama
```

### 5.2 Using Different LLM Providers

**Ollama (Local - Free):**

```bash
# First, ensure Ollama is running
ollama serve &

# Pull model if needed
ollama pull llama3.2

# Run experiment
python3 scripts/run_experiment.py \
    --llm-provider ollama \
    --llm-model llama3.2 \
    --dataset data/dataset_a.json
```

**OpenAI (Cloud):**

```bash
export OPENAI_API_KEY="your-key"

python3 scripts/run_experiment.py \
    --llm-provider openai \
    --llm-model gpt-4 \
    --dataset data/dataset_a.json
```

**Anthropic (Cloud):**

```bash
export ANTHROPIC_API_KEY="your-key"

python3 scripts/run_experiment.py \
    --llm-provider anthropic \
    --llm-model claude-3-5-sonnet-20241022 \
    --dataset data/dataset_a.json
```

### 5.3 Fast Mode for Production

**Enable Fast Mode** (4× faster, minimal quality loss):

```bash
python3 scripts/run_experiment.py \
    --fast-mode \
    --techniques baseline cot react \
    --llm-provider ollama
```

**Performance:**
- Standard Mode: ~8.2s/query, ~325 tokens
- Fast Mode: ~1.9s/query, ~78 tokens
- Accuracy: No change (100% → 100%)
- Quality Loss: -16% entropy (acceptable trade-off)

### 5.4 Custom Datasets

**Create Your Dataset:**

`my_dataset.json`:
```json
[
  {
    "question": "What is the capital of France?",
    "answer": "Paris",
    "category": "geography",
    "alternatives": ["paris"]
  },
  {
    "question": "What is 2 + 2?",
    "answer": "4",
    "category": "arithmetic"
  }
]
```

**Run with Custom Dataset:**

```bash
python3 scripts/run_experiment.py \
    --dataset my_dataset.json \
    --techniques baseline cot react
```

---

## 6. Understanding Results

### 6.1 Output Files

After running an experiment, you'll find:

```
results/
├── experiment_results.json      # Raw results (JSON)
├── experiment_report.md          # Human-readable report
├── visualization_report.md       # Visualization summary
└── figures/
    ├── improvement_over_baseline.png
    ├── accuracy_comparison_full.png
    ├── top_mistakes.png
    ├── metric_trends.png
    ├── entropy_distribution.png
    ├── perplexity_distribution.png
    └── ... (12 total visualizations)
```

### 6.2 Key Metrics Explained

**Accuracy**
- Definition: % of correct answers
- Range: 0-100%
- Higher is better
- Example: 85% = 85 out of 100 correct

**Entropy (H)**
- Definition: Model uncertainty in bits
- Formula: H = -Σ p(y|x) log₂ p(y|x)
- Range: 0+ (lower is better)
- Interpretation:
  - 0 = Perfectly confident
  - 1 = Slightly uncertain
  - 3+ = Highly uncertain

**Perplexity (PP)**
- Definition: Predictability measure
- Formula: PP = 2^H
- Range: 1+ (lower is better)
- Interpretation:
  - 1 = Perfect prediction
  - 4 = Moderate uncertainty
  - 10+ = High uncertainty

**Composite Loss (L)**
- Definition: Weighted combination of metrics
- Formula: L = 0.3·H + 0.2·|Y| + 0.2·PP + 0.3·(1-Acc)
- Range: 0+ (lower is better)
- Best for: Overall quality comparison

### 6.3 Reading the Report

**Section 1: Executive Summary**

```markdown
### Key Findings

**All techniques achieved 100% accuracy**, but quality differences emerged:

- **ReAct** (Reasoning and Acting) emerged as superior:
  - 18.51% reduction in loss
  - 33.65% reduction in entropy
  - Perfect cross-dataset consistency
```

**Interpretation:** ReAct is the best technique for quality, though all are equally accurate.

**Section 2: Technique Rankings**

```
Ranking by Composite Loss (Lower is Better):
1. 🥇 ReAct:        1.326 (-18.51% vs baseline)
2. 🥈 ToT:          1.342 (-17.52% vs baseline)
3. 🥉 CoT++:        1.368 (-15.91% vs baseline)
```

**Interpretation:** Use ReAct for highest quality, CoT for best cost-efficiency balance.

**Section 3: Statistical Significance**

```
Paired t-test Results:
- ReAct vs Baseline: t=-12.34, p<0.001 ✓✓✓ (highly significant)
- CoT vs Baseline:   t=-7.56,  p<0.001 ✓✓✓ (highly significant)
```

**Interpretation:** Improvements are statistically significant, not due to chance.

### 6.4 Visualization Guide

**Plot 1: Improvement Over Baseline**
- Shows % improvement for each technique
- Use to: Quickly identify best performers

**Plot 2: Entropy Distribution (Box Plot)**
- Shows uncertainty spread
- Use to: Assess consistency across samples

**Plot 3: Performance Heatmap**
- Shows all metrics side-by-side
- Use to: Compare techniques comprehensively

**Plot 4: Significance Matrix**
- Shows which comparisons are statistically significant
- Use to: Validate findings

---

## 7. Advanced Usage

### 7.1 Programmatic API

**Python Integration:**

```python
from pipeline import ExperimentConfig, ExperimentOrchestrator
from llm import OllamaClient

# Configure experiment
config = ExperimentConfig(
    dataset_paths={"dataset_a": "data/dataset_a.json"},
    llm_provider="ollama",
    llm_model="llama3.2",
    techniques=["baseline", "chain_of_thought", "react"],
    temperature=0.0,
    fast_mode=False,
)

# Run experiment
orchestrator = ExperimentOrchestrator(config)
results = orchestrator.run_experiment()

# Access results
print(f"Best technique: {results['best_technique']}")
print(f"Accuracy: {results['techniques']['react']['metrics']['accuracy']:.2%}")
```

**Using Building Blocks (NEW in v2.0):**

```python
from building_blocks import (
    JSONDataLoader,
    TechniquePromptBuilder,
    UnifiedLLMInterface,
    ComprehensiveMetricCalculator,
    ExperimentResultAggregator
)

# Load data
loader = JSONDataLoader()
dataset = loader.load("data/dataset_a.json")

# Build prompts
builder = TechniquePromptBuilder()
prompts = [builder.build(sample["question"], "chain_of_thought")
           for sample in dataset]

# Execute LLM calls
interface = UnifiedLLMInterface(provider="ollama", model="llama3.2")
responses = [interface.execute(prompt) for prompt in prompts]

# Calculate metrics
calculator = ComprehensiveMetricCalculator()
predictions = [r["response"] for r in responses]
ground_truths = [s["answer"] for s in dataset]
metrics = calculator.calculate(predictions, ground_truths)

# Aggregate results
aggregator = ExperimentResultAggregator()
results = [{"correct": p == g, **metrics}
           for p, g in zip(predictions, ground_truths)]
summary = aggregator.aggregate(results)

print(f"Accuracy: {summary['statistics']['accuracy']:.2%}")
```

**Parallel Execution (NEW in v2.0):**

```python
from pipeline.parallel import ParallelExecutor, parallel_evaluate_samples

# Create parallel executor
executor = ParallelExecutor(max_workers=4)

# Parallel evaluation
results = parallel_evaluate_samples(
    evaluate_func=my_evaluation_function,
    samples=dataset_samples,
    max_workers=4
)
```

### 7.2 Custom Prompt Techniques

**Define Your Own Technique:**

```python
from prompts.base import BasePromptBuilder, PromptTemplate, PromptTechnique

class MyCustomPrompt(BasePromptBuilder):
    def __init__(self):
        super().__init__(PromptTechnique.CUSTOM)

    def build(self, **kwargs) -> PromptTemplate:
        return PromptTemplate(
            technique=self.technique,
            system_prompt="You are an expert problem solver.",
            user_prompt="{question}\n\nSolve this carefully.",
            metadata={"description": "My custom technique"},
        )

# Use in experiment
from pipeline import ExperimentOrchestrator

orchestrator.add_technique("my_custom", MyCustomPrompt())
results = orchestrator.run_experiment()
```

### 7.3 Batch Processing

**Process Multiple Datasets:**

```python
import glob

datasets = glob.glob("data/*.json")

for dataset_path in datasets:
    config = ExperimentConfig(
        dataset_paths={"current": dataset_path},
        techniques=["baseline", "cot", "react"],
    )

    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.run_experiment()

    # Save results
    output_path = f"results/{Path(dataset_path).stem}_results.json"
    orchestrator.save_results(results, output_path)
```

### 7.4 Custom Metrics

**Add Your Own Metric:**

```python
from metrics.accuracy import evaluate_accuracy

def custom_metric(predicted: str, ground_truth: str) -> float:
    """Custom evaluation metric."""
    # Your logic here
    return similarity_score

# Integrate into pipeline
evaluator.add_metric("custom", custom_metric)
```

---

## 8. Troubleshooting

### 8.1 Common Issues

**Issue: "ModuleNotFoundError: No module named 'src'"**

Solution:
```bash
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install package
pip install -e .
```

**Issue: "OpenAI API Error: Incorrect API key"**

Solution:
```bash
# Check API key is set
echo $OPENAI_API_KEY

# Set if missing
export OPENAI_API_KEY="sk-..."
```

**Issue: "Ollama connection refused"**

Solution:
```bash
# Start Ollama server
ollama serve &

# Verify it's running
curl http://localhost:11434/api/tags
```

**Issue: "Coverage below 70%"**

Solution:
```bash
# Run tests with coverage
python3 -m pytest --cov=src --cov-report=html

# View detailed report
open htmlcov/index.html
```

### 8.2 Performance Issues

**Problem: Experiments taking too long**

Solutions:
1. Enable Fast Mode: `--fast-mode`
2. Reduce sample size: `--max-samples 50`
3. Use fewer techniques: `--techniques baseline cot`
4. Use local LLM (Ollama) instead of cloud APIs

**Problem: High memory usage**

Solutions:
1. Reduce batch size
2. Process datasets one at a time
3. Disable intermediate saves: `--no-save-intermediate`

### 8.3 Error Messages

**Error: "Dataset validation failed"**

Check dataset format:
```json
[
  {
    "question": "...",  ✓ Required
    "answer": "...",    ✓ Required
    "category": "..."   ✗ Optional
  }
]
```

**Error: "Insufficient test coverage (46%)"**

Run new tests:
```bash
python3 -m pytest tests/test_*.py --cov=src
```

---

## 9. FAQ

**Q: Which LLM provider should I use?**

A:
- **Ollama (local)**: Free, unlimited, privacy, sufficient quality
- **OpenAI GPT-4**: Highest quality, expensive
- **Anthropic Claude**: Good balance, moderate cost
- **Recommendation**: Start with Ollama for research, use cloud for production

**Q: Which technique is best?**

A:
- **Best Quality**: ReAct (-18.5% loss)
- **Best Efficiency**: Role-Based (4× faster than ReAct)
- **Best Balance**: Chain-of-Thought (good quality, reasonable cost)
- **Recommendation**: Use CoT for most cases, ReAct for critical applications

**Q: How many samples do I need?**

A:
- **Minimum**: 30 samples per technique (statistical validity)
- **Recommended**: 100+ samples (robust results)
- **Ideal**: 500+ samples (publication-quality)

**Q: Can I use this commercially?**

A: Yes! MIT License - free for commercial and academic use.

**Q: How do I cite this work?**

A:
```bibtex
@software{prompt_optimization_lab,
  author = {Barda, Tal},
  title = {Prompt Optimization \& Evaluation System},
  year = {2025},
  url = {https://github.com/TalBarda8/prompt-optimization-lab}
}
```

**Q: What's the difference between Standard and Fast Mode?**

A:
| Aspect | Standard Mode | Fast Mode |
|--------|---------------|-----------|
| Speed | 8.2s/query | 1.9s/query |
| Tokens | ~325 | ~78 |
| Quality | Best | 84% of standard |
| Use Case | Research | Production |

---

## 10. Support

### 10.1 Getting Help

**Documentation:**
- [README.md](../../README.md) - Overview
- [ARCHITECTURE.md](../architecture/ARCHITECTURE.md) - System design
- [PRD.md](../../PRD.md) - Requirements
- [PROMPT_ENGINEERING_LOG.md](../prompts/PROMPT_ENGINEERING_LOG.md) - Techniques

**Community:**
- GitHub Issues: https://github.com/TalBarda8/prompt-optimization-lab/issues
- Email: tal.barda@example.com

### 10.2 Reporting Bugs

**Bug Report Template:**

```markdown
**Environment:**
- OS: macOS 13.5
- Python: 3.10.8
- Version: 1.0.0

**Steps to Reproduce:**
1. Run `python3 scripts/run_experiment.py ...`
2. See error: ...

**Expected Behavior:**
...

**Actual Behavior:**
...

**Logs:**
```
[Paste error logs here]
```
```

### 10.3 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for contribution guidelines.

---

## Appendix A: Command Reference

### Complete CLI Options

```bash
python3 scripts/run_experiment.py \
    --config PATH              # Config file path
    --dataset PATH             # Dataset file path
    --llm-provider PROVIDER    # openai|anthropic|ollama
    --llm-model MODEL          # Model name
    --techniques LIST          # Space-separated list
    --temperature FLOAT        # 0.0-1.0
    --max-tokens INT           # Max output tokens
    --fast-mode                # Enable fast mode
    --output DIR               # Output directory
    --visualize                # Generate plots
    --save-intermediate        # Save per-sample results
    --random-seed INT          # Random seed
    --max-samples INT          # Limit sample count
    --verbose                  # Verbose logging
    --quiet                    # Minimal output
```

### Available Techniques

- `baseline` - Direct questioning
- `chain_of_thought` / `cot` - Step-by-step reasoning
- `chain_of_thought_plus_plus` / `cot_plus_plus` - CoT with verification
- `react` - Reasoning and Acting
- `tree_of_thoughts` / `tot` - Multi-path exploration
- `role_based` - Expert persona
- `few_shot` - Learning from examples

---

## Appendix B: Configuration Examples

### Minimal Configuration

```yaml
llm:
  provider: ollama
  model: llama3.2

experiment:
  techniques:
    - baseline
    - cot

datasets:
  main: data/dataset_a.json
```

### Full Configuration

```yaml
# LLM Configuration
llm:
  provider: openai
  model: gpt-4-turbo-preview
  temperature: 0.0
  max_tokens: 500
  timeout: 30

# Experiment Configuration
experiment:
  techniques:
    - baseline
    - chain_of_thought
    - chain_of_thought_plus_plus
    - react
    - tree_of_thoughts
    - role_based
    - few_shot
  fast_mode: false
  random_seed: 42
  max_samples: null  # Process all

# Dataset Configuration
datasets:
  dataset_a: data/dataset_a.json
  dataset_b: data/dataset_b.json

# Output Configuration
output:
  directory: results/
  save_intermediate: true
  generate_visualizations: true
  save_raw_responses: false

# Logging Configuration
logging:
  level: INFO
  file: logs/experiment.log
```

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-13 | Initial user manual | Tal Barda |
| 2.0 | 2025-12-15 | Added building blocks, multiprocessing, updated test counts (357 tests, 72% coverage) | Tal Barda |

---

**Project Statistics (v2.0):**
- **Total Tests**: 357 tests
- **Coverage**: 72%
- **Building Blocks**: 6 modular components
- **Modules**: 8 core containers
- **Multiprocessing**: Fully supported

---

**End of User Manual**
