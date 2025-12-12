"""
LLM Utilities

Token counting, response parsing, and logprobs extraction utilities.
"""

from typing import Dict, List, Any, Optional
import re
import tiktoken


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Input text
        model: Model name (for tokenizer selection)

    Returns:
        Number of tokens
    """
    try:
        # Get encoding for model
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Default to cl100k_base for GPT-4, GPT-3.5-turbo
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    return len(tokens)


def parse_response(response: Any, provider: str = "openai") -> str:
    """
    Parse LLM response to extract text content.

    Args:
        response: Raw API response
        provider: API provider ("openai", "anthropic", or "ollama")

    Returns:
        Extracted text content
    """
    if provider == "openai":
        # OpenAI response structure
        if hasattr(response, 'choices') and len(response.choices) > 0:
            return response.choices[0].message.content
        return str(response)

    elif provider == "anthropic":
        # Anthropic response structure
        if hasattr(response, 'content') and len(response.content) > 0:
            return response.content[0].text
        return str(response)

    elif provider == "ollama":
        # Ollama response structure (from subprocess)
        if isinstance(response, dict) and 'stdout' in response:
            return response['stdout'].strip()
        return str(response)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_logprobs(response: Any, provider: str = "openai") -> Optional[List[Dict[str, Any]]]:
    """
    Extract log probabilities from LLM response.

    For entropy and perplexity calculations (PRD Section 2.3).

    Args:
        response: Raw API response
        provider: API provider ("openai", "anthropic", or "ollama")

    Returns:
        List of logprobs dictionaries, or None if not available
    """
    if provider == "openai":
        # OpenAI logprobs structure
        if hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'logprobs') and choice.logprobs:
                # Extract token logprobs
                logprobs_data = []
                if hasattr(choice.logprobs, 'content') and choice.logprobs.content:
                    for token_logprob in choice.logprobs.content:
                        logprobs_data.append({
                            'token': token_logprob.token,
                            'logprob': token_logprob.logprob,
                            'bytes': getattr(token_logprob, 'bytes', None),
                        })
                return logprobs_data if logprobs_data else None
        return None

    elif provider == "anthropic":
        # Anthropic doesn't provide logprobs in the same way
        # Would need to use a different approach or API endpoint
        return None

    elif provider == "ollama":
        # Ollama CLI doesn't provide logprobs
        # Note: Ollama API (HTTP) may support logprobs in future versions
        return None

    else:
        raise ValueError(f"Unknown provider: {provider}")


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison.

    - Convert to lowercase
    - Remove extra whitespace
    - Remove punctuation at the end
    - Trim

    Args:
        answer: Raw answer string

    Returns:
        Normalized answer
    """
    # Convert to lowercase
    answer = answer.lower().strip()

    # Remove extra whitespace
    answer = ' '.join(answer.split())

    # Remove trailing punctuation
    answer = re.sub(r'[.,!?;:]+$', '', answer)

    return answer


def fuzzy_match(predicted: str, ground_truth: str, alternatives: List[str] = None) -> bool:
    """
    Fuzzy match predicted answer against ground truth.

    Checks:
    - Exact match (normalized)
    - Ground truth contained in prediction
    - Any alternative match

    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        alternatives: List of alternative acceptable answers

    Returns:
        True if match found
    """
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)

    # Exact match
    if pred_norm == gt_norm:
        return True

    # Ground truth contained in prediction
    if gt_norm in pred_norm:
        return True

    # Check alternatives
    if alternatives:
        for alt in alternatives:
            alt_norm = normalize_answer(alt)
            if pred_norm == alt_norm or alt_norm in pred_norm:
                return True

    return False
