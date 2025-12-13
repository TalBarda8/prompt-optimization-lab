"""
Prompt Optimization & Evaluation System

A comprehensive experimental framework for evaluating and optimizing LLM prompts
using information-theoretic metrics and statistical validation.

Author: Tal Barda
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Tal Barda"
__email__ = "tal.barda@example.com"

# Export key modules
from src import data
from src import llm
from src import prompts
from src import metrics
from src import visualization
from src import pipeline

__all__ = [
    "data",
    "llm",
    "prompts",
    "metrics",
    "visualization",
    "pipeline",
]
