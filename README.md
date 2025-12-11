# Prompt Optimization & Evaluation System

A comprehensive system for demonstrating measurable performance improvements through systematic prompt engineering using information-theoretic metrics and statistical validation.

## Overview

This project implements a rigorous experimental framework to:
- Evaluate 6+ prompt engineering techniques (CoT, CoT++, ReAct, ToT, Role-Based, Few-Shot)
- Measure improvements using entropy, perplexity, and accuracy metrics
- Validate results with statistical significance testing (p < 0.05)
- Generate publication-quality visualizations and reports

## Project Structure

```
prompt-optimization-lab/
├── README.md                    # This file
├── PRD.md                       # Complete Product Requirements Document
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variables template
├── config/
│   └── pipeline_config.yaml    # Pipeline configuration
├── data/
│   ├── dataset_a.json          # Simple QA dataset (75 samples)
│   └── dataset_b.json          # Multi-step reasoning dataset (35 samples)
├── src/
│   ├── data/                   # Dataset creation and validation
│   ├── prompts/                # Prompt engineering techniques
│   ├── evaluation/             # Metrics and statistical tests
│   ├── visualization/          # Graph generation
│   ├── llm/                    # LLM client with caching
│   └── pipeline/               # End-to-end orchestration
├── notebooks/                   # Jupyter notebooks for analysis
├── tests/                       # Unit and integration tests
├── results/                     # Experimental results
└── figures/                     # Generated visualizations
```

## Installation

### Prerequisites
- Python 3.9, 3.10, or 3.11
- OpenAI API key (for GPT-4) or Anthropic API key (for Claude)
- 8GB+ RAM recommended
- 10GB+ free disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/TalBarda8/prompt-optimization-lab.git
cd prompt-optimization-lab
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API key
```

## Usage

### Run the complete pipeline:
```bash
python main.py
```

### Run specific phases:
```bash
# Phase 1: Data Preparation
python main.py --phase data

# Phase 2: Baseline Evaluation
python main.py --phase baseline

# Phase 3: Prompt Optimization
python main.py --phase optimization

# Phase 4: Statistical Comparison
python main.py --phase evaluation

# Phase 5: Visualization
python main.py --phase visualization
```

### Interactive Analysis:
```bash
jupyter notebook notebooks/
```

## Configuration

Edit `config/pipeline_config.yaml` to customize:
- LLM provider and model
- Optimization techniques to test
- Loss function weights (α, β, γ, δ)
- Statistical test parameters

## Results

After running the pipeline, results will be available in:
- `results/baseline/` - Baseline evaluation metrics
- `results/optimized/` - Optimized prompt results
- `results/final_report.pdf` - Comprehensive analysis report
- `figures/` - All 12 required visualizations

## Key Metrics

The system evaluates prompts using:
- **Accuracy**: Task performance (with fuzzy matching)
- **Entropy H(Y|X)**: Output uncertainty (lower is better)
- **Perplexity**: Model confidence (lower is better)
- **Loss Function**: L = α·H + β·Length + γ·PPL + δ·(1-Acc)

## Success Criteria (from PRD)

- ✅ Statistically significant improvement (p < 0.05)
- ✅ Minimum 15% accuracy improvement over baseline
- ✅ 20%+ entropy reduction on average
- ✅ Publication-ready visualizations and documentation

## Testing

Run the test suite:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Documentation

- **PRD.md**: Complete product requirements specification
- **API Documentation**: See `docs/api.md` (generated)
- **Jupyter Notebooks**: Step-by-step analysis in `notebooks/`

## License

MIT License - See LICENSE file for details

## Citation

If you use this system in your research, please cite:

```bibtex
@software{prompt_optimization_2025,
  title={Prompt Optimization \& Evaluation System},
  author={AI Systems Engineering Team},
  year={2025},
  url={https://github.com/TalBarda8/prompt-optimization-lab}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the development team.

---

**Status**: 🚧 Under Development (Stage 0/13 Complete)

**Last Updated**: 2025-12-11
