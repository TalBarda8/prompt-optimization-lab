# Configuration Guide

**Project:** Prompt Optimization & Evaluation System
**Version:** 2.0.0
**Author:** Tal Barda
**Last Updated:** December 15, 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Configuration File Format](#2-configuration-file-format)
3. [LLM Configuration](#3-llm-configuration)
4. [Experiment Configuration](#4-experiment-configuration)
5. [Dataset Configuration](#5-dataset-configuration)
6. [Output Configuration](#6-output-configuration)
7. [Environment Variables](#7-environment-variables)
8. [Advanced Configuration](#8-advanced-configuration)
9. [Configuration Examples](#9-configuration-examples)
10. [Validation and Troubleshooting](#10-validation-and-troubleshooting)

---

## 1. Overview

### 1.1 Configuration Methods

The system supports three configuration methods (in order of precedence):

1. **Command-line arguments** (highest priority)
2. **Environment variables**
3. **Configuration file** (config.yaml)
4. **Default values** (lowest priority)

### 1.2 Configuration File Location

Default search paths (in order):
1. `./config.yaml` (current directory)
2. `./config/config.yaml`
3. `~/.prompt-opt/config.yaml` (user home directory)

Or specify explicitly:
```bash
python3 scripts/run_experiment.py --config /path/to/config.yaml
```

---

## 2. Configuration File Format

### 2.1 Basic Structure

The configuration file uses YAML format with five main sections:

```yaml
llm:           # LLM provider and model settings
experiment:    # Experiment parameters
datasets:      # Dataset paths and settings
output:        # Output and reporting options
logging:       # Logging configuration (optional)
```

### 2.2 Minimal Configuration

```yaml
llm:
  provider: ollama
  model: llama3.2

experiment:
  techniques:
    - baseline
    - chain_of_thought

datasets:
  dataset_a: data/dataset_a.json
```

### 2.3 Complete Configuration Template

```yaml
# ============================================================================
# LLM Configuration
# ============================================================================
llm:
  # Provider: openai | anthropic | ollama
  provider: ollama

  # Model name (provider-specific)
  model: llama3.2

  # Temperature (0.0 = deterministic, 1.0 = creative)
  temperature: 0.0

  # Maximum output tokens
  max_tokens: 500

  # Request timeout in seconds
  timeout: 30

  # Retry configuration
  max_retries: 3
  retry_delay: 1.0

# ============================================================================
# Experiment Configuration
# ============================================================================
experiment:
  # List of techniques to evaluate
  techniques:
    - baseline
    - chain_of_thought
    - chain_of_thought_plus_plus
    - react
    - tree_of_thoughts
    - role_based
    - few_shot

  # Fast mode (4× speedup with minimal quality loss)
  fast_mode: false

  # Random seed for reproducibility
  random_seed: 42

  # Maximum samples per dataset (null = all)
  max_samples: null

  # Baseline technique name (for comparisons)
  baseline_technique: baseline

  # Enable statistical validation
  run_statistical_tests: true

  # Significance level for tests
  alpha: 0.05

# ============================================================================
# Dataset Configuration
# ============================================================================
datasets:
  # Dataset name: file path
  dataset_a: data/dataset_a.json
  dataset_b: data/dataset_b.json

  # Validation settings
  validate_format: true
  require_categories: false
  allow_alternatives: true

# ============================================================================
# Output Configuration
# ============================================================================
output:
  # Output directory
  directory: results/

  # Save intermediate results per sample
  save_intermediate: true

  # Generate visualizations
  generate_visualizations: true

  # Save raw LLM responses
  save_raw_responses: false

  # Report format: markdown | html | both
  report_format: markdown

  # Figure format: png | pdf | svg
  figure_format: png

  # Figure DPI
  figure_dpi: 300

# ============================================================================
# Logging Configuration (Optional)
# ============================================================================
logging:
  # Log level: DEBUG | INFO | WARNING | ERROR
  level: INFO

  # Log file path
  file: logs/experiment.log

  # Console output
  console: true

  # Detailed logging
  verbose: false
```

---

## 3. LLM Configuration

### 3.1 Provider Selection

**OpenAI:**
```yaml
llm:
  provider: openai
  model: gpt-4-turbo-preview
  temperature: 0.0
  max_tokens: 500
```

**Anthropic:**
```yaml
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  temperature: 0.0
  max_tokens: 500
```

**Ollama (Local):**
```yaml
llm:
  provider: ollama
  model: llama3.2
  temperature: 0.0
  max_tokens: 500
```

### 3.2 Model Options

**OpenAI Models:**
- `gpt-4-turbo-preview` - Best quality, expensive
- `gpt-4` - High quality, expensive
- `gpt-3.5-turbo` - Good balance, affordable

**Anthropic Models:**
- `claude-3-opus-20240229` - Highest quality
- `claude-3-5-sonnet-20241022` - Best balance
- `claude-3-haiku-20240307` - Fastest, cheapest

**Ollama Models:**
- `llama3.2` - Recommended (3B params)
- `llama3.1` - Larger (8B params)
- `mistral` - Alternative
- `phi3` - Lightweight

### 3.3 Temperature Settings

```yaml
llm:
  # Deterministic (recommended for experiments)
  temperature: 0.0

  # Slightly creative
  # temperature: 0.3

  # Balanced
  # temperature: 0.5

  # Creative
  # temperature: 1.0
```

**Recommendations:**
- **Experiments**: 0.0 (reproducibility)
- **Production**: 0.0-0.3 (consistency)
- **Creative tasks**: 0.7-1.0 (diversity)

### 3.4 Token Limits

```yaml
llm:
  max_tokens: 500  # Default

  # For simple questions
  # max_tokens: 100

  # For detailed reasoning
  # max_tokens: 1000

  # For very long outputs
  # max_tokens: 2000
```

**Guidelines:**
- Baseline: 100-200 tokens
- CoT: 300-500 tokens
- ReAct/ToT: 500-1000 tokens

---

## 4. Experiment Configuration

### 4.1 Technique Selection

**Single Technique:**
```yaml
experiment:
  techniques:
    - chain_of_thought
```

**Multiple Techniques:**
```yaml
experiment:
  techniques:
    - baseline
    - chain_of_thought
    - react
```

**All Techniques:**
```yaml
experiment:
  techniques:
    - baseline
    - chain_of_thought
    - chain_of_thought_plus_plus
    - react
    - tree_of_thoughts
    - role_based
    - few_shot
```

**Available Technique Names:**
- `baseline`
- `chain_of_thought` (alias: `cot`)
- `chain_of_thought_plus_plus` (alias: `cot_plus_plus`, `cot++`)
- `react`
- `tree_of_thoughts` (alias: `tot`)
- `role_based`
- `few_shot`

### 4.2 Fast Mode

**Enable Fast Mode:**
```yaml
experiment:
  fast_mode: true  # 4× faster, 75% fewer tokens
```

**Performance Comparison:**

| Metric | Standard | Fast Mode |
|--------|----------|-----------|
| Speed | 8.2s/query | 1.9s/query |
| Tokens | ~325 | ~78 |
| Quality | 100% | 84% |

**When to Use:**
- ✅ Production deployments
- ✅ High-volume queries
- ✅ Cost-sensitive applications
- ✅ When accuracy is primary concern
- ❌ Research requiring maximum quality
- ❌ Complex multi-step reasoning

### 4.3 Reproducibility

**Set Random Seed:**
```yaml
experiment:
  random_seed: 42  # Any integer
```

**Benefits:**
- Identical results across runs
- Reproducible research
- Debugging consistency
- Fair A/B testing

**Note:** Only works with `temperature: 0.0`

### 4.4 Sample Limiting

**Process All Samples:**
```yaml
experiment:
  max_samples: null  # Default
```

**Limit for Testing:**
```yaml
experiment:
  max_samples: 10  # Quick test
```

**Limit for Development:**
```yaml
experiment:
  max_samples: 50  # Faster iteration
```

**Production:**
```yaml
experiment:
  max_samples: null  # Process all
```

### 4.5 Statistical Testing

**Enable Statistical Validation:**
```yaml
experiment:
  run_statistical_tests: true
  alpha: 0.05  # Significance level (95% confidence)
```

**Configure Tests:**
```yaml
experiment:
  run_statistical_tests: true
  alpha: 0.05
  use_bonferroni_correction: true  # Multiple comparison correction
  paired_tests: true  # Use paired t-tests
  wilcoxon_fallback: true  # Use Wilcoxon if normality fails
```

---

## 5. Dataset Configuration

### 5.1 Single Dataset

```yaml
datasets:
  main: data/dataset_a.json
```

### 5.2 Multiple Datasets

```yaml
datasets:
  dataset_a: data/dataset_a.json
  dataset_b: data/dataset_b.json
  custom: data/my_questions.json
```

### 5.3 Dataset Validation

```yaml
datasets:
  dataset_a: data/dataset_a.json

  # Validation settings
  validate_format: true  # Check JSON structure
  require_categories: false  # Category field optional
  allow_alternatives: true  # Allow alternative answers
```

### 5.4 Dataset Format

**Required Fields:**
```json
[
  {
    "question": "What is 2 + 2?",
    "answer": "4"
  }
]
```

**Recommended Fields:**
```json
[
  {
    "question": "What is 2 + 2?",
    "answer": "4",
    "category": "arithmetic",
    "alternatives": ["four"],
    "difficulty": "easy"
  }
]
```

---

## 6. Output Configuration

### 6.1 Output Directory

```yaml
output:
  directory: results/  # Default

  # Or custom path
  # directory: /path/to/output/

  # Or timestamped
  # directory: results/{timestamp}/
```

### 6.2 Intermediate Results

**Save Per-Sample Results:**
```yaml
output:
  save_intermediate: true  # Detailed per-sample data
```

**Benefits:**
- Debug individual predictions
- Analyze error patterns
- Resume interrupted experiments

**Disk Usage:**
- 100 samples: ~500 KB
- 1,000 samples: ~5 MB
- 10,000 samples: ~50 MB

### 6.3 Visualization Options

**Enable All Visualizations:**
```yaml
output:
  generate_visualizations: true
  figure_format: png
  figure_dpi: 300
```

**Configure Formats:**
```yaml
output:
  generate_visualizations: true

  # For presentations
  figure_format: png
  figure_dpi: 150

  # For publications
  # figure_format: pdf
  # figure_dpi: 300

  # For web
  # figure_format: svg
```

### 6.4 Report Generation

```yaml
output:
  report_format: markdown  # Default

  # Or HTML
  # report_format: html

  # Or both
  # report_format: both
```

### 6.5 Raw Response Saving

```yaml
output:
  save_raw_responses: false  # Default (saves space)

  # Enable for debugging
  # save_raw_responses: true
```

**When to Enable:**
- Debugging LLM outputs
- Analyzing response patterns
- Building training datasets
- Quality assurance

**Disk Impact:**
- Adds ~2-5× storage requirements

---

## 7. Environment Variables

### 7.1 API Keys

**OpenAI:**
```bash
export OPENAI_API_KEY="sk-..."
```

**Anthropic:**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 7.2 Configuration Overrides

**Fast Mode:**
```bash
export FAST_MODE=true
```

**Random Seed:**
```bash
export RANDOM_SEED=42
```

**Output Directory:**
```bash
export OUTPUT_DIR=results/
```

**Max Samples:**
```bash
export MAX_SAMPLES=100
```

**Log Level:**
```bash
export LOG_LEVEL=DEBUG
```

### 7.3 System Configuration

**Python Path:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**Ollama Host:**
```bash
export OLLAMA_HOST=http://localhost:11434
```

**GPU Settings:**
```bash
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

### 7.4 Loading from .env File

**Create .env file:**
```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
FAST_MODE=true
RANDOM_SEED=42
```

**Load automatically:**
```python
from dotenv import load_dotenv
load_dotenv()  # Loads .env file
```

---

## 8. Advanced Configuration

### 8.1 Custom Prompt Templates

**Configure Role-Based Prompts:**
```yaml
experiment:
  techniques:
    - role_based

  role_based_config:
    role: mathematician  # or scientist, teacher, expert
```

**Configure Few-Shot Prompts:**
```yaml
experiment:
  techniques:
    - few_shot

  few_shot_config:
    num_examples: 3
    example_source: dataset  # or custom
    custom_examples:
      - question: "What is 5 + 3?"
        answer: "8"
      - question: "What is 10 - 4?"
        answer: "6"
```

### 8.2 Metric Weights

**Customize Composite Loss:**
```yaml
experiment:
  loss_weights:
    entropy: 0.3      # Default: 0.3
    length: 0.2       # Default: 0.2
    perplexity: 0.2   # Default: 0.2
    accuracy: 0.3     # Default: 0.3
```

### 8.3 Parallel Processing (NEW in v2.0)

**Enable Multiprocessing:**

```yaml
experiment:
  parallel_execution: true
  max_workers: null  # Auto-detect (uses cpu_count() - 1)
  # Or specify manually:
  # max_workers: 4
```

**Performance Benefits:**
- Significant speedup for CPU-bound tasks
- Automatic optimal worker count
- Thread-safe execution
- Graceful error handling with fallback

**Configuration Options:**

```yaml
experiment:
  # Enable parallel processing
  parallel_execution: true

  # Worker configuration
  max_workers: null  # Auto (recommended)
  # max_workers: 4   # Manual override

  # Parallel evaluation modes
  parallel_samples: true      # Evaluate samples in parallel
  parallel_techniques: true   # Evaluate techniques in parallel
```

**Guidelines:**
- **CPU cores**: `max_workers: null` auto-detects optimal count
- **Memory**: ~500 MB per worker
- **API rate limits**: Respect provider limits (use rate limiting for cloud APIs)
- **Local LLMs**: Full parallelization supported
- **Cloud APIs**: Use sequential or limited parallelization

**Example: Full Parallelization**

```yaml
llm:
  provider: ollama  # Local model
  model: llama3.2

experiment:
  techniques:
    - baseline
    - chain_of_thought
    - react
  parallel_execution: true
  max_workers: null  # Auto-detect
```

**Example: Limited Parallelization for Cloud APIs**

```yaml
llm:
  provider: openai  # Cloud API
  model: gpt-4

experiment:
  techniques:
    - baseline
    - chain_of_thought
  parallel_execution: true
  max_workers: 2  # Limit to respect rate limits
```

### 8.4 Caching

```yaml
experiment:
  enable_caching: true
  cache_dir: .cache/
  cache_ttl: 86400  # 24 hours in seconds
```

**Benefits:**
- Skip duplicate queries
- Resume interrupted experiments
- Reduce API costs

---

## 9. Configuration Examples

### 9.1 Research Configuration

```yaml
# config/research.yaml
llm:
  provider: ollama
  model: llama3.2
  temperature: 0.0
  max_tokens: 1000

experiment:
  techniques:
    - baseline
    - chain_of_thought
    - chain_of_thought_plus_plus
    - react
    - tree_of_thoughts
  fast_mode: false  # Maximum quality
  random_seed: 42
  run_statistical_tests: true

datasets:
  dataset_a: data/dataset_a.json
  dataset_b: data/dataset_b.json

output:
  directory: results/research/
  save_intermediate: true
  generate_visualizations: true
  figure_format: pdf
  figure_dpi: 300
  report_format: both

logging:
  level: INFO
  file: logs/research.log
```

### 9.2 Production Configuration

```yaml
# config/production.yaml
llm:
  provider: openai
  model: gpt-3.5-turbo
  temperature: 0.0
  max_tokens: 200

experiment:
  techniques:
    - chain_of_thought  # Best balance
  fast_mode: true  # 4× faster
  max_samples: null
  run_statistical_tests: false

datasets:
  production: data/production_queries.json

output:
  directory: results/production/
  save_intermediate: false  # Save space
  generate_visualizations: false  # Skip plots
  save_raw_responses: false

logging:
  level: WARNING
  console: false
  file: logs/production.log
```

### 9.3 Development Configuration

```yaml
# config/dev.yaml
llm:
  provider: ollama
  model: llama3.2
  temperature: 0.0
  max_tokens: 500

experiment:
  techniques:
    - baseline
    - chain_of_thought
  fast_mode: true
  max_samples: 10  # Quick testing
  random_seed: 42

datasets:
  test: data/dataset_a.json

output:
  directory: results/dev/
  save_intermediate: true
  generate_visualizations: true

logging:
  level: DEBUG
  verbose: true
```

### 9.4 Cost-Optimized Configuration

```yaml
# config/budget.yaml
llm:
  provider: ollama  # Free local model
  model: llama3.2
  temperature: 0.0
  max_tokens: 100  # Minimal tokens

experiment:
  techniques:
    - baseline
    - role_based  # Most efficient
  fast_mode: true  # 4× token reduction
  max_samples: 50

datasets:
  main: data/dataset_a.json

output:
  directory: results/budget/
  save_intermediate: false
  generate_visualizations: false
  save_raw_responses: false
```

---

## 10. Validation and Troubleshooting

### 10.1 Validate Configuration

**Check Syntax:**
```bash
python3 -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

**Validate Schema:**
```bash
python3 scripts/validate_config.py config.yaml
```

### 10.2 Common Configuration Errors

**Error: "Invalid YAML syntax"**

```yaml
# Wrong (mixing tabs and spaces)
llm:
	provider: ollama
  model: llama3.2

# Correct (consistent indentation)
llm:
  provider: ollama
  model: llama3.2
```

**Error: "Unknown technique: cot_plusplus"**

```yaml
# Wrong
experiment:
  techniques:
    - cot_plusplus  # Underscore issue

# Correct
experiment:
  techniques:
    - chain_of_thought_plus_plus
    # or
    - cot_plus_plus
```

**Error: "Dataset file not found"**

```yaml
# Wrong (relative to config file)
datasets:
  main: dataset.json

# Correct (absolute or relative to execution directory)
datasets:
  main: data/dataset.json
  # or
  main: /absolute/path/to/dataset.json
```

### 10.3 Configuration Precedence

**Example:**
```bash
# config.yaml has: fast_mode: false
# Environment has: FAST_MODE=true
# Command-line has: --fast-mode

# Result: Command-line wins (--fast-mode)
```

**Precedence Order:**
1. Command-line arguments
2. Environment variables
3. Configuration file
4. Default values

### 10.4 Debug Configuration

**Print Active Configuration:**
```bash
python3 scripts/run_experiment.py --print-config
```

**Dry Run:**
```bash
python3 scripts/run_experiment.py --dry-run --config config.yaml
```

**Verbose Logging:**
```bash
python3 scripts/run_experiment.py --verbose --config config.yaml
```

---

## Appendix A: Configuration Schema

### Full YAML Schema

```yaml
# Type: object
llm:
  provider: string  # enum: openai, anthropic, ollama
  model: string
  temperature: float  # range: 0.0-1.0
  max_tokens: integer  # range: 1-4096
  timeout: integer  # seconds
  max_retries: integer
  retry_delay: float

experiment:
  techniques: array[string]
  fast_mode: boolean
  random_seed: integer
  max_samples: integer | null
  baseline_technique: string
  run_statistical_tests: boolean
  alpha: float  # range: 0.0-1.0

datasets:
  [name: string]: path: string

output:
  directory: string
  save_intermediate: boolean
  generate_visualizations: boolean
  save_raw_responses: boolean
  report_format: string  # enum: markdown, html, both
  figure_format: string  # enum: png, pdf, svg
  figure_dpi: integer

logging:
  level: string  # enum: DEBUG, INFO, WARNING, ERROR
  file: string
  console: boolean
  verbose: boolean
```

---

## Appendix B: Environment Variable Reference

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | string | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | string | - | Anthropic API key |
| `OLLAMA_HOST` | string | `http://localhost:11434` | Ollama server URL |
| `FAST_MODE` | boolean | `false` | Enable fast mode |
| `RANDOM_SEED` | integer | `42` | Random seed |
| `OUTPUT_DIR` | string | `results/` | Output directory |
| `MAX_SAMPLES` | integer | - | Sample limit |
| `LOG_LEVEL` | string | `INFO` | Log level |

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2025-12-13 | Initial configuration guide | Tal Barda |
| 2.0 | 2025-12-15 | Added multiprocessing configuration, updated for v2.0 | Tal Barda |

---

**What's New in v2.0:**
- **Multiprocessing Support**: Complete parallel processing configuration
- **Building Blocks**: Modular architecture (no new config needed)
- **Enhanced Testing**: 357 tests with 72% coverage
- **Improved Performance**: Automatic worker optimization

---

**End of Configuration Guide**
