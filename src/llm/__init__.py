"""
LLM Client Module

Provides unified interface for LLM API calls (OpenAI, Anthropic, Ollama).
Handles token counting, response parsing, and logprobs extraction.
Supports local LLM inference via Ollama.
"""

from .client import LLMClient, LLMResponse
from .utils import count_tokens, parse_response, extract_logprobs

__all__ = [
    "LLMClient",
    "LLMResponse",
    "count_tokens",
    "parse_response",
    "extract_logprobs",
]
