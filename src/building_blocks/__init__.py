"""
Building Blocks Module

Defines core building blocks for the Prompt Optimization & Evaluation System.

This module implements the Building Blocks design pattern (Chapter 17 compliance),
providing:
- Clear input/output contracts
- Single responsibility components
- Reusable, composable building blocks
- Well-defined interfaces

Building Blocks:
1. DataLoader: Load and validate datasets
2. PromptBuilder: Construct prompts from techniques
3. LLMInterface: Execute LLM calls
4. MetricCalculator: Calculate evaluation metrics
5. ResultAggregator: Aggregate and summarize results
6. Visualizer: Generate visualizations
"""

from .interfaces import (
    BuildingBlock,
    BuildingBlockContract,
    DataLoaderBlock,
    PromptBuilderBlock,
    LLMInterfaceBlock,
    MetricCalculatorBlock,
    ResultAggregatorBlock,
    VisualizerBlock
)

from .implementations import (
    JSONDataLoader,
    TechniquePromptBuilder,
    UnifiedLLMInterface,
    ComprehensiveMetricCalculator,
    ExperimentResultAggregator,
    MatplotlibVisualizer
)

__all__ = [
    # Interfaces
    "BuildingBlock",
    "BuildingBlockContract",
    "DataLoaderBlock",
    "PromptBuilderBlock",
    "LLMInterfaceBlock",
    "MetricCalculatorBlock",
    "ResultAggregatorBlock",
    "VisualizerBlock",
    # Implementations
    "JSONDataLoader",
    "TechniquePromptBuilder",
    "UnifiedLLMInterface",
    "ComprehensiveMetricCalculator",
    "ExperimentResultAggregator",
    "MatplotlibVisualizer",
]
