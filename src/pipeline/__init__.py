"""
Pipeline Module

Orchestrates the complete experimental workflow.
"""

from .orchestrator import ExperimentOrchestrator, ExperimentConfig
from .evaluator import BaselineEvaluator, PromptOptimizationEvaluator
from .statistics import StatisticalValidator
from .experiment_evaluator import evaluate_technique, collect_dataset_results
from .summary import print_experiment_summary, generate_summary_dict
from .parallel import (
    ParallelExecutor,
    parallel_evaluate_samples,
    parallel_evaluate_techniques
)

__all__ = [
    "ExperimentOrchestrator",
    "ExperimentConfig",
    "BaselineEvaluator",
    "PromptOptimizationEvaluator",
    "StatisticalValidator",
    "evaluate_technique",
    "collect_dataset_results",
    "print_experiment_summary",
    "generate_summary_dict",
    "ParallelExecutor",
    "parallel_evaluate_samples",
    "parallel_evaluate_techniques",
]
