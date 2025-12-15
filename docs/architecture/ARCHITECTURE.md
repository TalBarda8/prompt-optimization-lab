# System Architecture Documentation

**Project**: Prompt Optimization & Evaluation System
**Version**: 2.0.0
**Last Updated**: 2025-12-15
**Author**: Tal Barda

---

## Table of Contents

1. [Overview](#overview)
2. [System Context (C4 Level 1)](#system-context)
3. [Container Architecture (C4 Level 2)](#container-architecture)
4. [Component Architecture (C4 Level 3)](#component-architecture)
5. [Deployment Architecture](#deployment-architecture)
6. [Data Flow](#data-flow)
7. [Technology Stack](#technology-stack)
8. [Design Patterns](#design-patterns)
9. [API Interfaces](#api-interfaces)
10. [Configuration Management](#configuration-management)

---

## Overview

The Prompt Optimization & Evaluation System is a comprehensive experimental framework designed for academic research in LLM prompt engineering. The system evaluates 7 prompt engineering techniques across 140 carefully crafted samples using information-theoretic metrics and statistical validation.

**Core Capabilities**:
- Multi-provider LLM integration (OpenAI, Anthropic, Ollama)
- Information-theoretic metric computation (Entropy, Perplexity, Loss)
- Statistical significance testing (t-tests, Wilcoxon, Bonferroni)
- Automated visualization generation (12 publication-ready charts)
- End-to-end pipeline orchestration

---

## System Context (C4 Level 1)

```
┌─────────────────────────────────────────────────────────────┐
│                   Prompt Optimization System                  │
│                                                               │
│  Evaluates & compares prompt engineering techniques using    │
│  mathematical metrics and statistical validation              │
└─────────────────────────────────────────────────────────────┘
                              │
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
  ┌───────────────┐   ┌───────────────┐   ┌──────────────┐
  │  OpenAI API   │   │Anthropic API  │   │    Ollama    │
  │   (GPT-4)     │   │   (Claude)    │   │  (Local LLM) │
  └───────────────┘   └───────────────┘   └──────────────┘

  External Systems:
  - LLM Providers: Generate responses for evaluation
  - File System: Dataset storage and results persistence
  - User: Researchers/developers running experiments
```

**Stakeholders**:
- **Primary**: Academic researchers studying prompt engineering
- **Secondary**: NLP engineers optimizing LLM applications
- **Tertiary**: Students learning prompt engineering techniques

---

## Container Architecture (C4 Level 2)

The system is organized into 8 main containers (Python packages):

```
┌──────────────────────────────────────────────────────────────┐
│                    Application Container                      │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐            │
│  │    Data    │  │    LLM     │  │   Prompts   │            │
│  │  Module    │  │  Client    │  │   Module    │            │
│  └────────────┘  └────────────┘  └─────────────┘            │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌─────────────┐            │
│  │  Metrics   │  │Visualization│  │   Pipeline  │            │
│  │  Module    │  │   Module    │  │ Orchestrator│            │
│  └────────────┘  └────────────┘  └─────────────┘            │
│                                                               │
│  ┌────────────┐  ┌──────────────────────────────┐            │
│  │  Building  │  │        Parallel Executor      │            │
│  │   Blocks   │  │   (Multiprocessing Support)   │            │
│  └────────────┘  └──────────────────────────────┘            │
│                                                               │
│  ┌────────────────────────────────────────────────┐          │
│  │            CLI & Main Entry Point              │          │
│  └────────────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────────┘
         │                  │                   │
         ▼                  ▼                   ▼
    ┌────────┐       ┌────────────┐      ┌──────────┐
    │Dataset │       │Configuration│      │ Results  │
    │ Files  │       │    YAML     │      │  JSON    │
    └────────┘       └────────────┘      └──────────┘
```

### Container Responsibilities

1. **Data Module**: Dataset creation, loading, validation
2. **LLM Client**: Multi-provider API abstraction
3. **Prompts Module**: 7 prompt technique implementations
4. **Metrics Module**: Information-theoretic calculations
5. **Visualization Module**: Chart generation (12 types)
6. **Pipeline Module**: Orchestration, statistics, summarization, parallel execution
7. **Building Blocks Module**: Modular composable components (6 building blocks)
8. **Parallel Executor**: Multiprocessing support for performance optimization

---

## Component Architecture (C4 Level 3)

### Data Module Components

```
src/data/
├── dataset_creator.py    # Creates Complex QA & Reasoning datasets
├── loaders.py            # JSON dataset loading
├── validators.py         # PRD compliance validation
└── __init__.py           # Module exports
```

**Key Classes**:
- `DatasetCreator`: Generates 140 samples across 2 datasets
- `DatasetValidator`: Validates samples against PRD specifications
- `load_dataset()`: Loads and parses JSON datasets

### LLM Client Components

```
src/llm/
├── client.py             # Multi-provider LLM client (with auto-download)
├── utils.py              # Token counting, response parsing
└── __init__.py
```

**Key Classes**:
- `LLMClient`: Unified interface for OpenAI, Anthropic, Ollama
  - **Auto-Download**: Automatically downloads missing Ollama models via `_ensure_ollama_model_exists()`
  - **Fast Mode**: Optimized token limits and timeouts for local LLMs
  - **Streaming Progress**: Real-time download progress via subprocess streaming
- `ResponseParser`: Extracts answers and log probabilities
- `TokenCounter`: Estimates token usage for cost tracking

**Design Pattern**: **Adapter Pattern** - Provides unified interface to different LLM APIs

**Default Provider**: Ollama (local, free, auto-downloads models as needed)

### Prompts Module Components

```
src/prompts/
├── base.py               # Base prompt template class
├── techniques.py         # 7 technique implementations
├── techniques/           # Individual technique modules (if refactored)
└── __init__.py
```

**Key Classes**:
- `PromptTechnique` (base class)
- `BaselinePrompt`, `ChainOfThoughtPrompt`, `CoTPlusPlus`
- `ReactPrompt`, `TreeOfThoughtsPrompt`, `RoleBasedPrompt`, `FewShotPrompt`

**Design Pattern**: **Strategy Pattern** - Interchangeable prompt techniques

### Metrics Module Components

```
src/metrics/
├── accuracy.py           # Accuracy calculations
├── information_theory.py # Entropy, perplexity, loss
└── __init__.py
```

**Key Classes**:
- `AccuracyCalculator`: Exact match and fuzzy matching
- `InformationTheoryMetrics`: Entropy, perplexity, composite loss
- `MetricAggregator`: Combines metrics across samples

### Visualization Module Components

```
src/visualization/
├── plotters.py           # 12 chart generation functions
├── visualization.py      # High-level viz orchestration
├── report.py             # Markdown/HTML report generation
└── __init__.py
```

**Key Functions**:
- `plot_accuracy_comparison()`, `plot_loss_comparison()`
- `plot_entropy_distribution()`, `plot_heatmap()`
- `generate_all_visualizations()`: Produces all 12 charts

### Pipeline Module Components

```
src/pipeline/
├── orchestrator.py       # Main pipeline coordinator
├── evaluator.py          # Single technique evaluation
├── experiment_evaluator.py # Multi-technique experiments
├── statistics.py         # Statistical validation
├── summary.py            # Results summarization
├── parallel.py           # Multiprocessing support (NEW)
└── __init__.py
```

**Key Classes**:
- `ExperimentOrchestrator`: Coordinates full experimental pipeline
- `TechniqueEvaluator`: Evaluates one technique on dataset
- `StatisticalValidator`: t-tests, Wilcoxon, Bonferroni correction
- `ExperimentSummarizer`: Rich console output and JSON export
- `ParallelExecutor`: Thread-safe parallel processing (NEW)

**Design Pattern**: **Facade Pattern** - Orchestrator provides simplified interface to complex subsystem

### Building Blocks Module Components (NEW)

```
src/building_blocks/
├── interfaces.py         # Abstract building block interfaces
├── implementations.py    # Concrete implementations
└── __init__.py
```

**Key Interfaces**:
- `DataLoaderBlock`: Load and validate datasets
- `PromptBuilderBlock`: Construct prompts from techniques
- `LLMInterfaceBlock`: Execute LLM calls
- `MetricCalculatorBlock`: Calculate evaluation metrics
- `ResultAggregatorBlock`: Aggregate experiment results
- `VisualizerBlock`: Generate visualizations

**Design Pattern**: **Building Blocks Pattern** - Modular composable components with clear contracts

---

## Deployment Architecture

### Local Development Environment

```
┌────────────────────────────────────────┐
│    Developer Machine (macOS/Linux)     │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │  Python 3.9+ Virtual Environment │ │
│  │                                  │ │
│  │  ┌────────────────────────────┐ │ │
│  │  │  Application Runtime       │ │ │
│  │  │  - CLI Entry Point         │ │ │
│  │  │  - Pipeline Execution      │ │ │
│  │  └────────────────────────────┘ │ │
│  │                                  │ │
│  │  ┌────────────────────────────┐ │ │
│  │  │  Local File System         │ │ │
│  │  │  - data/                   │ │ │
│  │  │  - results/                │ │ │
│  │  │  - cache/                  │ │ │
│  │  └────────────────────────────┘ │ │
│  └──────────────────────────────────┘ │
│                                        │
│        │              │                │
│        │              │                │
└────────┼──────────────┼────────────────┘
         │              │
         ▼              ▼
   ┌──────────┐   ┌──────────────┐
   │Cloud LLM │   │ Ollama Service│
   │   APIs   │   │  (localhost)  │
   └──────────┘   └──────────────┘
```

### Production/CI Environment

```
┌─────────────────────────────────────┐
│       GitHub Actions CI/CD          │
│                                     │
│  1. Checkout Code                   │
│  2. Setup Python 3.9                │
│  3. Install Dependencies            │
│  4. Run Unit Tests (pytest)         │
│  5. Generate Coverage Report        │
│  6. Run Integration Tests (optional)│
│  7. Build Documentation             │
│                                     │
└─────────────────────────────────────┘
```

---

## Data Flow

### Complete Experiment Flow

```
1. Load Configuration
   └─> config/pipeline_config.yaml

2. Load Datasets
   └─> data/dataset_a.json (Complex QA - 90 samples)
   └─> data/dataset_b.json (Multi-step Reasoning - 50 samples)
   └─> Validation against PRD specifications

3. Initialize LLM Client
   └─> Select provider (openai/anthropic/ollama)
   └─> Authenticate with API key or local service
   └─> Configure temperature, max_tokens

4. For Each Technique:
   a. Apply Technique
      └─> Generate prompts for all 140 samples
      └─> Send to LLM client
      └─> Collect responses with logprobs

   b. Calculate Metrics
      └─> Accuracy (exact match + fuzzy)
      └─> Entropy H(Y|X) = -Σ p(y|x) log₂ p(y|x)
      └─> Perplexity = 2^H(Y|X)
      └─> Composite Loss L(P,D)

   c. Store Results
      └─> results/[technique]_results.json

5. Statistical Analysis
   └─> Paired t-tests (technique vs. baseline)
   └─> Wilcoxon signed-rank tests
   └─> Bonferroni correction (α/n comparisons)
   └─> Calculate 95% confidence intervals

6. Generate Visualizations
   └─> 12 publication-ready charts (PNG + PDF)
   └─> results/figures/

7. Produce Summary
   └─> Rich console output
   └─> JSON experiment results
   └─> Markdown report
```

---

## Technology Stack

### Core Python Packages

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| **Scientific** | `numpy` | 1.24+ | Array operations |
| | `pandas` | 2.0+ | Data manipulation |
| | `scipy` | 1.10+ | Statistical tests |
| **LLM Integration** | `openai` | 1.0+ | OpenAI API |
| | `anthropic` | 0.18+ | Claude API |
| | `tiktoken` | 0.5+ | Token counting |
| **Statistics** | `statsmodels` | 0.14+ | Advanced statistics |
| | `scikit-learn` | 1.3+ | ML utilities |
| **Visualization** | `matplotlib` | 3.7+ | Base plotting |
| | `seaborn` | 0.12+ | Statistical viz |
| | `plotly` | 5.14+ | Interactive charts |
| **Testing** | `pytest` | 7.4+ | Unit testing |
| | `pytest-cov` | 4.1+ | Coverage reports |
| **CLI/UX** | `rich` | 13.0+ | Console formatting |
| | `tqdm` | 4.65+ | Progress bars |
| **Config** | `pyyaml` | 6.0+ | YAML parsing |
| | `python-dotenv` | 1.0+ | Environment vars |

### External Services

- **OpenAI API**: GPT-4 model access
- **Anthropic API**: Claude model access
- **Ollama**: Local LLM inference (llama3.2, mistral, phi3)

---

## Design Patterns

### 1. Strategy Pattern (Prompts)

**Context**: Need to switch between 7 different prompt techniques
**Solution**: Define common interface, implement variants

```python
class PromptTechnique(ABC):
    @abstractmethod
    def apply(self, question: str) -> str:
        pass

class ChainOfThoughtPrompt(PromptTechnique):
    def apply(self, question: str) -> str:
        return f"Let's think step by step.\n{question}"
```

### 2. Adapter Pattern (LLM Client)

**Context**: Multiple LLM providers with different APIs
**Solution**: Unified `LLMClient` adapts to each provider

```python
class LLMClient:
    def __init__(self, provider: str):
        if provider == "openai":
            self.client = OpenAI()
        elif provider == "anthropic":
            self.client = Anthropic()
        elif provider == "ollama":
            self.client = OllamaClient()

    def complete(self, prompt: str) -> Response:
        # Adapts to provider-specific API
```

### 3. Facade Pattern (Pipeline Orchestrator)

**Context**: Complex subsystem interactions
**Solution**: Single orchestrator coordinates all modules

```python
class ExperimentOrchestrator:
    def run_experiment(self, config):
        # Simplified interface to complex pipeline
        data = self.data_module.load()
        results = self.llm_module.evaluate(data)
        metrics = self.metrics_module.calculate(results)
        self.viz_module.generate(metrics)
```

### 4. Template Method Pattern (Evaluators)

**Context**: Common evaluation flow with variant steps
**Solution**: Base evaluator with hook methods

```python
class BaseEvaluator(ABC):
    def evaluate(self, dataset):
        self.prepare()
        results = self.execute(dataset)
        self.finalize(results)
        return results
```

### 5. Builder Pattern (Configuration)

**Context**: Complex experiment configuration
**Solution**: Fluent configuration builder

```python
config = (ExperimentConfig()
    .with_model("gpt-4")
    .with_techniques(["baseline", "cot"])
    .with_datasets(["dataset_a", "dataset_b"])
    .build())
```

---

## API Interfaces

### Internal Module Interfaces

#### Data Module API

```python
# Loading
dataset = load_dataset("data/dataset_a.json")  # -> List[Sample]

# Validation
is_valid = validate_dataset(dataset, prd_spec)  # -> bool

# Creation
creator = DatasetCreator()
creator.generate_complex_qa(n=90)  # -> List[Sample]
```

#### LLM Client API

```python
client = LLMClient(provider="openai", model="gpt-4")

# Single completion
response = client.complete(
    prompt="What is 2+2?",
    temperature=0.0,
    max_tokens=100
)  # -> Response(text, logprobs, tokens)

# Batch completion
responses = client.batch_complete(prompts)  # -> List[Response]
```

#### Metrics API

```python
# Accuracy
acc = calculate_accuracy(predictions, ground_truth)  # -> float

# Information theory
entropy = calculate_entropy(logprobs)  # -> float
perplexity = calculate_perplexity(entropy)  # -> float
loss = calculate_composite_loss(metrics, weights)  # -> float
```

#### Visualization API

```python
# Generate single chart
plot_accuracy_comparison(results, output="figures/accuracy.png")

# Generate all charts
generate_all_visualizations(
    results=experiment_results,
    output_dir="results/figures"
)  # -> List[Path]
```

### External API Interactions

#### OpenAI API

```http
POST https://api.openai.com/v1/chat/completions
Content-Type: application/json
Authorization: Bearer {API_KEY}

{
  "model": "gpt-4",
  "messages": [{"role": "user", "content": "..."}],
  "temperature": 0.0,
  "max_tokens": 100,
  "logprobs": true,
  "top_logprobs": 5
}
```

#### Ollama Local API

```http
POST http://localhost:11434/api/generate
Content-Type: application/json

{
  "model": "llama3.2",
  "prompt": "...",
  "stream": false,
  "options": {
    "temperature": 0.0,
    "num_predict": 100
  }
}
```

---

## Configuration Management

### Configuration Files

1. **`config/pipeline_config.yaml`**: Main experiment configuration
2. **`.env`**: API keys and secrets (not committed)
3. **`pyproject.toml`**: Package metadata and tool configuration

### Configuration Schema

```yaml
# pipeline_config.yaml

model:
  provider: "ollama"        # ollama (default) | openai | anthropic
  model_name: "phi3"        # Model identifier (auto-downloaded if missing)
  temperature: 0.0          # Sampling temperature
  max_tokens: 500           # Max response length
  timeout: 60               # Request timeout (seconds)
  fast_mode: false          # Enable fast mode for local LLMs (2-5x speedup)

datasets:
  dataset_a: "data/dataset_a.json"
  dataset_b: "data/dataset_b.json"
  validation_strict: true

techniques:
  enabled:
    - baseline
    - chain_of_thought
    - chain_of_thought_plus_plus
    - react
    - tree_of_thoughts
    - role_based
    - few_shot

evaluation:
  loss_function_weights:
    alpha: 0.3     # Entropy weight
    beta: 0.2      # Length weight
    gamma: 0.2     # Perplexity weight
    delta: 0.3     # Accuracy weight

  statistical_tests:
    alpha: 0.05    # Significance level
    bonferroni_correction: true

visualization:
  output_dir: "results/figures"
  formats: ["png", "pdf"]
  dpi: 300
  style: "seaborn-v0_8"

parallel:
  enabled: false            # Enable multiprocessing
  max_workers: null         # null = auto (cpu_count - 1)
  chunk_size: 10            # Samples per worker

output:
  results_dir: "results"
  cache_dir: "cache"
  log_dir: "logs"
```

### Environment Variables

```bash
# .env (not committed to git)

# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Optional
OPENAI_ORG_ID=org-...
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

---

## Security Considerations

### API Key Management

1. **Never commit secrets**: `.env` is in `.gitignore`
2. **Environment variables**: All secrets loaded via `python-dotenv`
3. **Key rotation**: Support for multiple API keys
4. **Example template**: `.env.example` provided without real keys

### Data Privacy

1. **No PII in datasets**: All samples are synthetic
2. **Local-first option**: Ollama for offline experiments
3. **No telemetry**: No analytics or usage tracking

### Dependency Security

1. **Pinned versions**: `requirements.txt` specifies exact versions
2. **Regular updates**: Dependencies updated monthly
3. **Vulnerability scanning**: `pip-audit` recommended

---

## Future Architecture Considerations

### Scalability

1. **Distributed Processing**: Add Celery/RQ for task queue
2. **Database Backend**: PostgreSQL for large result storage
3. **Caching**: Redis for API response caching

### Extensibility

1. **Plugin System**: Load custom techniques dynamically
2. **Custom Metrics**: User-defined metric plugins
3. **Export Formats**: Add CSV, Excel, LaTeX table exports

### Monitoring

1. **Logging**: Structured logging with `structlog`
2. **Metrics**: Prometheus-compatible metrics export
3. **Tracing**: OpenTelemetry integration for debugging

---

## References

1. **C4 Model**: https://c4model.com/
2. **Design Patterns**: Gang of Four (GoF) patterns
3. **Python Best Practices**: PEP 8, PEP 484 (Type Hints)
4. **Academic Guidelines**: Dr. Segal Yoram, Software Submission Guidelines v2.0

---

**Project Statistics**:
- **Version**: 2.0.0
- **Total Tests**: 357 (72% coverage)
- **Modules**: 8 core containers
- **Building Blocks**: 6 composable components
- **Prompt Techniques**: 7 implemented
- **Test Coverage**: 72%

**Document Status**: ✅ Complete
**Last Review**: 2025-12-15
**Next Review**: 2026-01-15
