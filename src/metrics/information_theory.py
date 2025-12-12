"""
Information-Theoretic Metrics

Implements mathematical formulas from PRD Section 2.3:
- Entropy H(Y|X)
- Perplexity
- Loss Function L(P, D)
"""

from typing import List, Dict, Any, Optional
import numpy as np
import math


def entropy_from_logprobs(logprobs: List[Dict[str, Any]]) -> float:
    """
    Calculate entropy from token log probabilities.

    Formula (PRD 2.3.1):
        H(Y|X) = -Σ p(y|x) log₂ p(y|x)

    Where:
        - p(y|x) = probability of token y given context x
        - logprob = log(p(y|x)) (natural log from API)

    Args:
        logprobs: List of logprob dictionaries from LLM response
                  Each dict should have 'token' and 'logprob' keys

    Returns:
        Entropy in bits (using log₂)
    """
    if not logprobs:
        return 0.0

    total_entropy = 0.0

    for token_data in logprobs:
        # Get log probability (natural log from API)
        logprob = token_data.get('logprob', 0.0)

        # Convert to probability: p = e^(logprob)
        probability = math.exp(logprob)

        # Avoid log(0)
        if probability > 0:
            # Calculate entropy: -p * log₂(p)
            # Convert natural log to log₂: log₂(p) = log(p) / log(2)
            log2_prob = logprob / math.log(2)
            total_entropy += -probability * log2_prob

    return total_entropy


def calculate_entropy(
    logprobs: Optional[List[Dict[str, Any]]] = None,
    probabilities: Optional[List[float]] = None
) -> float:
    """
    Calculate entropy from either logprobs or probabilities.

    PRD Formula (Section 2.3.1):
        H(Y|X) = -Σ p(y|x) log₂ p(y|x)

    Args:
        logprobs: List of logprob dictionaries
        probabilities: List of probabilities (alternative input)

    Returns:
        Entropy in bits
    """
    if logprobs is not None:
        return entropy_from_logprobs(logprobs)

    elif probabilities is not None:
        total_entropy = 0.0
        for prob in probabilities:
            if prob > 0:
                total_entropy += -prob * math.log2(prob)
        return total_entropy

    else:
        raise ValueError("Must provide either logprobs or probabilities")


def calculate_perplexity(entropy: float) -> float:
    """
    Calculate perplexity from entropy.

    PRD Formula (Section 2.3.2):
        Perplexity = 2^H(Y|X)

    Where:
        - H(Y|X) is the entropy in bits

    Lower perplexity indicates more confident/deterministic responses.

    Args:
        entropy: Entropy value in bits

    Returns:
        Perplexity value
    """
    return 2 ** entropy


def calculate_perplexity_from_logprobs(logprobs: List[Dict[str, Any]]) -> float:
    """
    Calculate perplexity directly from logprobs.

    Combines entropy calculation and perplexity formula.

    Args:
        logprobs: List of logprob dictionaries

    Returns:
        Perplexity value
    """
    entropy = entropy_from_logprobs(logprobs)
    return calculate_perplexity(entropy)


def calculate_loss(
    entropy: float,
    response_length: int,
    perplexity: float,
    accuracy: float,
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.2,
    delta: float = 0.3,
    normalize: bool = True,
) -> float:
    """
    Calculate composite loss function.

    PRD Formula (Section 2.3.3):
        L(P, D) = α·H(Y|X) + β·|Y| + γ·Perplexity + δ·(1 - Accuracy)

    Where:
        - α = 0.3: Weight for entropy (uncertainty)
        - β = 0.2: Weight for response length
        - γ = 0.2: Weight for perplexity
        - δ = 0.3: Weight for accuracy
        - Weights sum to 1.0

    Lower loss is better (indicates better prompt).

    Args:
        entropy: Entropy H(Y|X) in bits
        response_length: Number of tokens in response |Y|
        perplexity: Perplexity value
        accuracy: Accuracy score (0.0 to 1.0)
        alpha: Weight for entropy
        beta: Weight for length
        gamma: Weight for perplexity
        delta: Weight for accuracy
        normalize: Whether to normalize components to [0, 1] range

    Returns:
        Composite loss value
    """
    # Validate weights sum to 1.0
    total_weight = alpha + beta + gamma + delta
    if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
        raise ValueError(
            f"Weights must sum to 1.0, got {total_weight}. "
            f"alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}"
        )

    if normalize:
        # Normalize each component to [0, 1] range for fair weighting
        # These normalization constants are approximate based on expected ranges

        # Entropy: typically 0-10 bits for natural language
        entropy_norm = min(entropy / 10.0, 1.0)

        # Length: normalize by typical max length (e.g., 500 tokens)
        length_norm = min(response_length / 500.0, 1.0)

        # Perplexity: typically 1-100 for good models
        perplexity_norm = min(math.log(perplexity) / math.log(100), 1.0)

        # Accuracy: already in [0, 1]
        accuracy_norm = 1.0 - accuracy

        # Calculate normalized loss
        loss = (
            alpha * entropy_norm +
            beta * length_norm +
            gamma * perplexity_norm +
            delta * accuracy_norm
        )

    else:
        # Raw loss (no normalization)
        loss = (
            alpha * entropy +
            beta * response_length +
            gamma * perplexity +
            delta * (1.0 - accuracy)
        )

    return loss


def calculate_average_entropy(samples_logprobs: List[List[Dict[str, Any]]]) -> float:
    """
    Calculate average entropy across multiple samples.

    Args:
        samples_logprobs: List of logprobs lists (one per sample)

    Returns:
        Average entropy
    """
    if not samples_logprobs:
        return 0.0

    entropies = [entropy_from_logprobs(lp) for lp in samples_logprobs if lp]
    return np.mean(entropies) if entropies else 0.0


def calculate_average_perplexity(samples_logprobs: List[List[Dict[str, Any]]]) -> float:
    """
    Calculate average perplexity across multiple samples.

    Args:
        samples_logprobs: List of logprobs lists (one per sample)

    Returns:
        Average perplexity
    """
    if not samples_logprobs:
        return 0.0

    perplexities = [calculate_perplexity_from_logprobs(lp) for lp in samples_logprobs if lp]
    return np.mean(perplexities) if perplexities else 0.0


# ============================================================================
# FALLBACK METRICS FOR MODELS WITHOUT LOGPROBS (e.g., Ollama)
# ============================================================================

def calculate_fallback_entropy(
    predictions: List[str],
    accuracy: float,
    response_lengths: List[int]
) -> float:
    """
    Calculate approximate entropy when logprobs are not available.

    Uses response diversity and accuracy as proxy:
    - Low accuracy + high diversity → high entropy (uncertain)
    - High accuracy + low diversity → low entropy (confident)

    Args:
        predictions: List of predicted answers
        accuracy: Overall accuracy score (0.0 to 1.0)
        response_lengths: List of response lengths in characters

    Returns:
        Approximate entropy in bits (0-10 range)
    """
    if not predictions:
        return 5.0  # Default moderate entropy

    # Calculate response diversity (normalized unique answers ratio)
    unique_predictions = len(set(pred.strip().lower() for pred in predictions))
    total_predictions = len(predictions)
    diversity = unique_predictions / total_predictions if total_predictions > 0 else 0.5

    # Calculate length variance (normalized)
    if response_lengths:
        length_std = np.std(response_lengths)
        avg_length = np.mean(response_lengths)
        length_variance = (length_std / avg_length) if avg_length > 0 else 0.5
    else:
        length_variance = 0.5

    # Entropy estimate: high when accuracy is low OR diversity is high
    # Scale to typical entropy range (0-10 bits)
    base_entropy = (1.0 - accuracy) * 6.0  # Uncertainty from errors
    diversity_entropy = diversity * 4.0     # Uncertainty from diversity
    variance_entropy = length_variance * 2.0  # Uncertainty from variance

    entropy = (base_entropy + diversity_entropy + variance_entropy) / 3.0

    # Clip to reasonable range
    return min(max(entropy, 0.1), 10.0)


def calculate_fallback_perplexity(entropy: float) -> float:
    """
    Calculate perplexity from fallback entropy.

    Same formula as standard: Perplexity = 2^H

    Args:
        entropy: Estimated entropy in bits

    Returns:
        Perplexity value
    """
    return calculate_perplexity(entropy)


def calculate_fallback_loss(
    accuracy: float,
    avg_response_length: int,
    entropy: float,
    perplexity: float,
    alpha: float = 0.3,
    beta: float = 0.2,
    gamma: float = 0.2,
    delta: float = 0.3,
) -> float:
    """
    Calculate loss using fallback metrics.

    Uses the same formula as standard loss calculation.

    Args:
        accuracy: Overall accuracy (0.0 to 1.0)
        avg_response_length: Average response length
        entropy: Fallback entropy estimate
        perplexity: Fallback perplexity estimate
        alpha: Weight for entropy
        beta: Weight for length
        gamma: Weight for perplexity
        delta: Weight for accuracy

    Returns:
        Composite loss value
    """
    return calculate_loss(
        entropy=entropy,
        response_length=avg_response_length,
        perplexity=perplexity,
        accuracy=accuracy,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        normalize=True,
    )
