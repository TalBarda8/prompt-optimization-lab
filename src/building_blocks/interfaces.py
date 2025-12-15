"""
Building Block Interfaces

Defines abstract interfaces for all building blocks following the
Single Responsibility Principle and clear input/output contracts.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BuildingBlockContract:
    """
    Defines the contract for a building block.

    Attributes:
        name: Name of the building block
        input_schema: Expected input format
        output_schema: Guaranteed output format
        dependencies: Other building blocks this depends on
    """
    name: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    dependencies: List[str]


class BuildingBlock(ABC):
    """
    Base interface for all building blocks.

    Every building block must:
    - Have a clear contract
    - Implement process() method
    - Be stateless (or manage state explicitly)
    - Handle errors gracefully
    """

    @property
    @abstractmethod
    def contract(self) -> BuildingBlockContract:
        """Return the contract for this building block."""
        pass

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        Process input and return output according to contract.

        Args:
            input_data: Input conforming to input_schema

        Returns:
            Output conforming to output_schema
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """Validate input against contract."""
        return True  # Override for strict validation

    def validate_output(self, output_data: Any) -> bool:
        """Validate output against contract."""
        return True  # Override for strict validation


class DataLoaderBlock(BuildingBlock):
    """
    Building block for loading and validating datasets.

    Input: File path or dataset identifier
    Output: Validated dataset (List[Dict])

    Responsibilities:
    - Load data from source
    - Validate data format
    - Handle errors
    """

    @abstractmethod
    def load(self, source: str) -> List[Dict[str, Any]]:
        """Load dataset from source."""
        pass

    @abstractmethod
    def validate(self, dataset: List[Dict[str, Any]]) -> bool:
        """Validate dataset format."""
        pass


class PromptBuilderBlock(BuildingBlock):
    """
    Building block for constructing prompts.

    Input: Question/task and technique specification
    Output: Formatted prompt template

    Responsibilities:
    - Apply prompt engineering technique
    - Format prompt correctly
    - Include examples if needed
    """

    @abstractmethod
    def build(self, question: str, technique: str, **kwargs) -> Any:
        """Build prompt for given question and technique."""
        pass


class LLMInterfaceBlock(BuildingBlock):
    """
    Building block for LLM interactions.

    Input: Prompt template and parameters
    Output: LLM response with metadata

    Responsibilities:
    - Execute LLM call
    - Extract response
    - Handle rate limiting
    - Manage tokens
    """

    @abstractmethod
    def execute(self, prompt: Any, **params) -> Dict[str, Any]:
        """Execute LLM call and return response."""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class MetricCalculatorBlock(BuildingBlock):
    """
    Building block for metric calculation.

    Input: Predictions and ground truth
    Output: Calculated metrics

    Responsibilities:
    - Calculate accuracy
    - Calculate entropy/perplexity
    - Calculate loss
    - Return standardized metrics
    """

    @abstractmethod
    def calculate(
        self,
        predictions: List[str],
        ground_truths: List[str],
        logprobs: Optional[List[Any]] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Calculate all relevant metrics."""
        pass


class ResultAggregatorBlock(BuildingBlock):
    """
    Building block for aggregating results.

    Input: Individual evaluation results
    Output: Aggregated statistics and summaries

    Responsibilities:
    - Aggregate metrics across samples
    - Calculate statistics
    - Generate summaries
    """

    @abstractmethod
    def aggregate(
        self,
        results: List[Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """Aggregate results into summary statistics."""
        pass


class VisualizerBlock(BuildingBlock):
    """
    Building block for generating visualizations.

    Input: Aggregated results
    Output: Generated visualizations (figures/plots)

    Responsibilities:
    - Create visualizations
    - Save to files
    - Maintain consistent styling
    """

    @abstractmethod
    def visualize(
        self,
        data: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> str:
        """Generate visualization and return path to saved file."""
        pass


__all__ = [
    "BuildingBlock",
    "BuildingBlockContract",
    "DataLoaderBlock",
    "PromptBuilderBlock",
    "LLMInterfaceBlock",
    "MetricCalculatorBlock",
    "ResultAggregatorBlock",
    "VisualizerBlock",
]
