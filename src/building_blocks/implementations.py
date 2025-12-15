"""
Building Block Implementations

Concrete implementations of building block interfaces that wrap
existing functionality into the building blocks pattern.
"""

from typing import Any, Dict, List, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from building_blocks.interfaces import (
    BuildingBlockContract,
    DataLoaderBlock,
    PromptBuilderBlock,
    LLMInterfaceBlock,
    MetricCalculatorBlock,
    ResultAggregatorBlock,
    VisualizerBlock
)

# Import existing modules
from data.loaders import load_dataset
from prompts.techniques import (
    BaselinePrompt,
    ChainOfThoughtPrompt,
    ChainOfThoughtPlusPlusPrompt,
    ReActPrompt,
    TreeOfThoughtsPrompt,
    RoleBasedPrompt,
    FewShotPrompt
)
from llm.client import LLMClient
from llm.utils import count_tokens
from metrics.accuracy import calculate_dataset_accuracy
from metrics.information_theory import (
    calculate_average_entropy,
    calculate_average_perplexity,
    calculate_loss
)


class JSONDataLoader(DataLoaderBlock):
    """Loads datasets from JSON files."""

    @property
    def contract(self) -> BuildingBlockContract:
        return BuildingBlockContract(
            name="JSONDataLoader",
            input_schema={"source": "str (file path)"},
            output_schema={"dataset": "List[Dict[str, Any]]"},
            dependencies=[]
        )

    def process(self, input_data: str) -> List[Dict[str, Any]]:
        """Load and validate dataset from JSON file."""
        dataset = self.load(input_data)
        if self.validate(dataset):
            return dataset
        raise ValueError(f"Invalid dataset format: {input_data}")

    def load(self, source: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        data = load_dataset(source)
        # Handle both dict and list formats
        if isinstance(data, dict) and "samples" in data:
            return data["samples"]
        elif isinstance(data, list):
            return data
        return []

    def validate(self, dataset: List[Dict[str, Any]]) -> bool:
        """Validate dataset has required fields."""
        if not dataset:
            return True  # Empty is valid
        return all("question" in item and "answer" in item for item in dataset)


class TechniquePromptBuilder(PromptBuilderBlock):
    """Builds prompts using various techniques."""

    def __init__(self):
        self.builders = {
            "baseline": BaselinePrompt(),
            "chain_of_thought": ChainOfThoughtPrompt(),
            "chain_of_thought_plus_plus": ChainOfThoughtPlusPlusPrompt(),
            "react": ReActPrompt(),
            "tree_of_thoughts": TreeOfThoughtsPrompt(),
            "role_based": RoleBasedPrompt(),
            "few_shot": FewShotPrompt()
        }

    @property
    def contract(self) -> BuildingBlockContract:
        return BuildingBlockContract(
            name="TechniquePromptBuilder",
            input_schema={
                "question": "str",
                "technique": "str",
                "fast_mode": "bool (optional)"
            },
            output_schema={"prompt_template": "PromptTemplate"},
            dependencies=[]
        )

    def process(self, input_data: Dict[str, Any]) -> Any:
        """Build prompt from input data."""
        question = input_data["question"]
        technique = input_data.get("technique", "baseline")
        # Extract kwargs (everything except question and technique)
        kwargs = {k: v for k, v in input_data.items() if k not in ["question", "technique"]}
        return self.build(question, technique, **kwargs)

    def build(self, question: str, technique: str, **kwargs) -> Any:
        """Build prompt using specified technique."""
        builder = self.builders.get(technique, self.builders["baseline"])
        return builder.build(question, **kwargs)


class UnifiedLLMInterface(LLMInterfaceBlock):
    """Unified interface for LLM interactions."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider
        self.model = model
        try:
            self.client = LLMClient(provider=provider, model=model)
        except Exception:
            self.client = None  # Handle initialization errors

    @property
    def contract(self) -> BuildingBlockContract:
        return BuildingBlockContract(
            name="UnifiedLLMInterface",
            input_schema={
                "prompt": "PromptTemplate",
                "temperature": "float (optional)",
                "max_tokens": "int (optional)"
            },
            output_schema={
                "response": "str",
                "tokens": "int",
                "logprobs": "Optional[List[Dict]]"
            },
            dependencies=["PromptBuilder"]
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute LLM call from input data."""
        prompt = input_data["prompt"]
        params = {k: v for k, v in input_data.items() if k != "prompt"}
        return self.execute(prompt, **params)

    def execute(self, prompt: Any, **params) -> Dict[str, Any]:
        """Execute LLM call and return structured response."""
        if self.client is None:
            return {
                "response": "",
                "tokens": 0,
                "logprobs": None
            }

        try:
            response = self.client.generate(prompt, **params)
            return {
                "response": response.get("text", ""),
                "tokens": response.get("tokens", 0),
                "logprobs": response.get("logprobs")
            }
        except Exception:
            return {"response": "", "tokens": 0, "logprobs": None}

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return count_tokens(text, model=self.model)


class ComprehensiveMetricCalculator(MetricCalculatorBlock):
    """Calculates all evaluation metrics."""

    @property
    def contract(self) -> BuildingBlockContract:
        return BuildingBlockContract(
            name="ComprehensiveMetricCalculator",
            input_schema={
                "predictions": "List[str]",
                "ground_truths": "List[str]",
                "logprobs": "Optional[List[Any]]",
                "response_lengths": "Optional[List[int]]"
            },
            output_schema={
                "accuracy": "float",
                "entropy": "float",
                "perplexity": "float",
                "loss": "float"
            },
            dependencies=[]
        )

    def process(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate metrics from input data."""
        return self.calculate(**input_data)

    def calculate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        logprobs: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate all metrics."""
        # Accuracy
        accuracy_result = calculate_dataset_accuracy(predictions, ground_truths)
        accuracy = accuracy_result.get("accuracy", 0.0)

        # Entropy and Perplexity
        if logprobs:
            entropy = calculate_average_entropy(logprobs)
            perplexity = calculate_average_perplexity(logprobs)
        else:
            entropy = 0.0
            perplexity = 1.0

        # Loss
        response_lengths = kwargs.get("response_lengths", [len(p) for p in predictions])
        avg_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

        loss = calculate_loss(
            entropy=entropy,
            response_length=avg_length,
            perplexity=perplexity,
            accuracy=accuracy
        )

        return {
            "accuracy": accuracy,
            "entropy": entropy,
            "perplexity": perplexity,
            "loss": loss
        }


class ExperimentResultAggregator(ResultAggregatorBlock):
    """Aggregates experiment results."""

    @property
    def contract(self) -> BuildingBlockContract:
        return BuildingBlockContract(
            name="ExperimentResultAggregator",
            input_schema={"results": "List[Dict[str, Any]]"},
            output_schema={
                "summary": "Dict[str, Any]",
                "statistics": "Dict[str, float]"
            },
            dependencies=["MetricCalculator"]
        )

    def process(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results."""
        return self.aggregate(input_data)

    def aggregate(
        self,
        results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Aggregate individual results into summary."""
        if not results:
            return {"summary": {}, "statistics": {}}

        # Calculate aggregate statistics
        total = len(results)
        correct = sum(1 for r in results if r.get("correct", False))

        metrics = {
            "total_samples": total,
            "correct_count": correct,
            "accuracy": correct / total if total > 0 else 0.0
        }

        # Aggregate other metrics if present
        for key in ["entropy", "perplexity", "loss", "tokens"]:
            values = [r[key] for r in results if key in r]
            if values:
                metrics[f"avg_{key}"] = sum(values) / len(values)
                metrics[f"total_{key}"] = sum(values)

        return {
            "summary": {"technique": results[0].get("technique", "unknown")},
            "statistics": metrics
        }


class MatplotlibVisualizer(VisualizerBlock):
    """Generates visualizations using Matplotlib."""

    @property
    def contract(self) -> BuildingBlockContract:
        return BuildingBlockContract(
            name="MatplotlibVisualizer",
            input_schema={
                "data": "Dict[str, Any]",
                "plot_type": "str",
                "output_path": "str"
            },
            output_schema={"file_path": "str"},
            dependencies=["ResultAggregator"]
        )

    def process(self, input_data: Dict[str, Any]) -> str:
        """Generate visualization."""
        data = input_data["data"]
        output_path = input_data["output_path"]
        return self.visualize(data, output_path, **input_data)

    def visualize(
        self,
        data: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> str:
        """Generate and save visualization."""
        # This is a stub - actual visualization would use matplotlib/seaborn
        # For now, just return the path
        return output_path


__all__ = [
    "JSONDataLoader",
    "TechniquePromptBuilder",
    "UnifiedLLMInterface",
    "ComprehensiveMetricCalculator",
    "ExperimentResultAggregator",
    "MatplotlibVisualizer",
]
