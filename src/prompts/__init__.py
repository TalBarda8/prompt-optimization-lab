"""
Prompt Engineering Module

Implements 6+ prompt optimization techniques from PRD Section 3.
"""

from .base import PromptTemplate, PromptTechnique
from .techniques import (
    BaselinePrompt,
    ChainOfThoughtPrompt,
    ChainOfThoughtPlusPlusPrompt,
    ReActPrompt,
    TreeOfThoughtsPrompt,
    RoleBasedPrompt,
    FewShotPrompt,
)

__all__ = [
    # Base classes
    "PromptTemplate",
    "PromptTechnique",
    # Techniques
    "BaselinePrompt",
    "ChainOfThoughtPrompt",
    "ChainOfThoughtPlusPlusPrompt",
    "ReActPrompt",
    "TreeOfThoughtsPrompt",
    "RoleBasedPrompt",
    "FewShotPrompt",
]
