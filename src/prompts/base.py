"""
Base Prompt Template Classes

Provides foundation for prompt engineering techniques.
"""

from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass


class PromptTechnique(Enum):
    """Enumeration of prompt techniques."""
    BASELINE = "baseline"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CHAIN_OF_THOUGHT_PLUS_PLUS = "chain_of_thought_plus_plus"
    REACT = "react"
    TREE_OF_THOUGHTS = "tree_of_thoughts"
    ROLE_BASED = "role_based"
    FEW_SHOT = "few_shot"


@dataclass
class PromptTemplate:
    """
    Base prompt template.

    Attributes:
        technique: Prompt technique used
        system_prompt: System-level instruction (optional)
        user_prompt: User-level prompt
        examples: Few-shot examples (optional)
        metadata: Additional metadata
    """
    technique: PromptTechnique
    system_prompt: Optional[str] = None
    user_prompt: str = ""
    examples: Optional[List[Dict[str, str]]] = None
    metadata: Optional[Dict[str, Any]] = None

    def format(self, question: str, **kwargs) -> str:
        """
        Format the prompt with a specific question.

        Args:
            question: The question/problem to solve
            **kwargs: Additional formatting parameters

        Returns:
            Formatted prompt string
        """
        # Default implementation: just insert question
        return self.user_prompt.format(question=question, **kwargs)

    def get_full_prompt(self, question: str, **kwargs) -> Dict[str, str]:
        """
        Get full prompt with system and user parts.

        Args:
            question: The question/problem
            **kwargs: Additional parameters

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        return {
            "system": self.system_prompt or "",
            "user": self.format(question, **kwargs),
        }


class BasePromptBuilder:
    """Base class for building prompt templates."""

    def __init__(self, technique: PromptTechnique):
        """
        Initialize prompt builder.

        Args:
            technique: Prompt technique to use
        """
        self.technique = technique

    def build(self, **kwargs) -> PromptTemplate:
        """
        Build a prompt template.

        Args:
            **kwargs: Configuration parameters

        Returns:
            PromptTemplate instance
        """
        raise NotImplementedError("Subclasses must implement build()")

    def validate(self, template: PromptTemplate) -> bool:
        """
        Validate a prompt template.

        Args:
            template: Template to validate

        Returns:
            True if valid
        """
        if not template.user_prompt:
            return False
        if template.technique != self.technique:
            return False
        return True
