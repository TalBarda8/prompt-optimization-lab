"""
Experiment Orchestrator

Main pipeline coordinator that runs the complete experimental workflow.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data import load_dataset
from llm import LLMClient
from prompts import (
    BaselinePrompt,
    ChainOfThoughtPrompt,
    ChainOfThoughtPlusPlusPrompt,
    ReActPrompt,
    TreeOfThoughtsPrompt,
    RoleBasedPrompt,
    FewShotPrompt,
)
from .summary import (
    print_experiment_summary,
    generate_summary_dict,
)

# Import phase-specific modules
from .orchestrator_data import load_datasets
from .orchestrator_evaluation import (
    run_baseline_evaluation,
    run_prompt_optimization,
    calculate_all_metrics,
    run_statistical_validation,
)
from .orchestrator_reporting import (
    generate_visualizations,
    save_results,
)


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment orchestration.

    Attributes:
        dataset_paths: Dict mapping dataset names to file paths
        llm_provider: LLM provider ("openai", "anthropic", or "ollama")
        llm_model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        techniques: List of prompt technique names to evaluate
        output_dir: Directory for results
        save_intermediate: Whether to save intermediate results
        fast_mode: Enable fast mode (shorter prompts, reduced timeouts for Ollama)
    """
    dataset_paths: Dict[str, str] = field(default_factory=dict)
    llm_provider: str = "ollama"
    llm_model: str = "phi3"
    temperature: float = 0.0
    max_tokens: int = 500
    techniques: List[str] = field(default_factory=list)
    output_dir: str = "results"
    save_intermediate: bool = True
    fast_mode: bool = False


class ExperimentOrchestrator:
    """
    Orchestrates the complete experimental pipeline.

    Phases (PRD Section 5):
    1. Dataset Loading
    2. Baseline Evaluation
    3. Prompt Optimization
    4. Metric Calculation
    5. Statistical Validation
    6. Visualization Generation
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize orchestrator.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.llm_client = LLMClient(
            provider=config.llm_provider,
            model=config.llm_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            fast_mode=config.fast_mode,
        )

        # Create output directory
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize technique builders
        self.technique_builders = {
            "baseline": BaselinePrompt(),
            "chain_of_thought": ChainOfThoughtPrompt(),
            "chain_of_thought_plus_plus": ChainOfThoughtPlusPlusPrompt(),
            "react": ReActPrompt(),
            "tree_of_thoughts": TreeOfThoughtsPrompt(),
            "role_based": RoleBasedPrompt(),
            "few_shot": FewShotPrompt(),
        }

        # Results storage
        self.results = {
            "config": {
                "llm_provider": config.llm_provider,
                "llm_model": config.llm_model,
                "temperature": config.temperature,
                "techniques": config.techniques,
            },
            "datasets": {},
            "techniques": {},
            "baseline_technique": "baseline",
            "statistical_tests": {},
            "summary": {},
            "metadata": {
                "start_time": None,
                "end_time": None,
                "total_samples": 0,
                "total_api_calls": 0,
                "uses_fallback_metrics": False,
            },
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete experimental pipeline.

        Returns:
            Dictionary containing all results
        """
        self.results["metadata"]["start_time"] = datetime.now().isoformat()

        self._print_header()
        self._apply_fast_mode_filtering()

        # Phase 1: Load Datasets
        print("\n[Phase 1/6] Loading Datasets...")
        load_datasets(self.config.dataset_paths, self.results)

        # Phase 2: Baseline Evaluation
        print("\n[Phase 2/6] Running Baseline Evaluation...")
        run_baseline_evaluation(self.config.techniques, self)

        # Phase 3: Prompt Optimization
        print("\n[Phase 3/6] Running Prompt Optimization...")
        run_prompt_optimization(self.config.techniques, self)

        # Phase 4: Metric Calculation
        print("\n[Phase 4/6] Calculating Metrics...")
        calculate_all_metrics(self.results)

        # Phase 5: Statistical Validation
        print("\n[Phase 5/6] Statistical Validation...")
        run_statistical_validation(self.results)

        # Phase 6: Generate Visualizations
        print("\n[Phase 6/6] Generating Visualizations...")
        generate_visualizations(self.results, self.config.techniques, self.output_path)

        # Save final results
        save_results(self.results, self.output_path)

        self.results["metadata"]["end_time"] = datetime.now().isoformat()
        self.results["summary"] = generate_summary_dict(self.results)

        # Print human-readable summary
        print_experiment_summary(
            self.results,
            model_name=self.config.llm_model,
            show_top_mistakes=True,
            top_k=5,
        )

        print(f"\n📁 Results saved to: {self.output_path}/experiment_results.json")
        print(f"📊 Visualizations: {self.output_path}/figures/")
        print(f"🔢 Total API calls: {self.results['metadata']['total_api_calls']}")

        if self.results["metadata"]["uses_fallback_metrics"]:
            print("\n⚠️  Note: Fallback metrics used (model doesn't provide logprobs)")

        return self.results

    def _print_header(self):
        """Print pipeline header."""
        print("=" * 70)
        print("PROMPT OPTIMIZATION EXPERIMENTAL PIPELINE")
        print("=" * 70)
        print(f"Model: {self.config.llm_model}")
        print(f"Provider: {self.config.llm_provider}")

        if self.config.fast_mode:
            print("🚀 FAST MODE: ENABLED")
            print("  • Shortened prompts for all techniques")
            print("  • Reduced timeouts (20s vs 60s normal)")
            print("  • Token limits reduced (16 vs 32 normal)")

            if self.config.llm_provider == "ollama" and "llama3.2" in self.config.llm_model:
                print("  ⚠️  TIP: Consider using 'phi3' for faster inference:")
                print("      ollama pull phi3")
                print("      python main.py run-experiment --provider ollama --model phi3 --fast-mode")

    def _apply_fast_mode_filtering(self):
        """Apply fast mode filtering to techniques."""
        if self.config.fast_mode:
            heavy_techniques = ["tree_of_thoughts", "chain_of_thought_plus_plus"]
            skipped = [t for t in heavy_techniques if t in self.config.techniques]
            self.config.techniques = [t for t in self.config.techniques if t not in heavy_techniques]

            if skipped:
                print(f"  ⏭️  Skipping heavy techniques: {', '.join(skipped)}")

        print(f"Techniques: {', '.join(self.config.techniques)}")
        print(f"Output: {self.config.output_dir}")
        print("=" * 70)

    def get_technique_builder(self, technique_name: str):
        """
        Get prompt builder for a technique.

        Args:
            technique_name: Name of technique

        Returns:
            Prompt builder instance
        """
        return self.technique_builders.get(technique_name)
