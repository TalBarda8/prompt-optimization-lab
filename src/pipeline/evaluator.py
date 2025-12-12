"""
Prompt Evaluator

Evaluates prompt techniques on datasets and collects metrics.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm import LLMClient, LLMResponse
from llm.utils import fuzzy_match
from metrics import (
    calculate_entropy,
    calculate_perplexity,
    calculate_loss,
    calculate_accuracy,
    entropy_from_logprobs,
)
from prompts import PromptTemplate


class BaselineEvaluator:
    """
    Evaluates baseline prompt technique.

    Simple direct questioning without optimization.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize evaluator.

        Args:
            llm_client: LLM client instance
        """
        self.llm_client = llm_client

    def evaluate_dataset(
        self,
        dataset: Dict[str, Any],
        prompt_template: PromptTemplate,
        save_responses: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate baseline on a dataset.

        Args:
            dataset: Dataset dictionary
            prompt_template: Prompt template to use
            save_responses: Whether to save individual responses

        Returns:
            Evaluation results dictionary
        """
        samples = dataset.get("samples", [])
        dataset_type = dataset.get("dataset_type", "unknown")

        results = {
            "dataset_type": dataset_type,
            "total_samples": len(samples),
            "responses": [] if save_responses else None,
            "metrics": {
                "accuracies": [],
                "entropies": [],
                "perplexities": [],
                "losses": [],
                "response_lengths": [],
            },
            "summary": {},
        }

        print(f"    Evaluating {len(samples)} samples...")

        for i, sample in enumerate(samples):
            # Get question/problem
            if dataset_type == "simple_qa":
                question = sample["question"]
                ground_truth = sample["ground_truth"]
                alternatives = sample.get("alternative_answers", [])
            else:  # multi_step_reasoning
                question = sample["problem"]
                ground_truth = sample["ground_truth_solution"]["final_answer"]
                alternatives = []

            # Format prompt
            full_prompt = prompt_template.get_full_prompt(question)

            # Generate response
            response = self.llm_client.generate(
                prompt=full_prompt["user"],
                system_prompt=full_prompt["system"] if full_prompt["system"] else None,
                logprobs=True,
            )

            # Calculate accuracy
            accuracy = calculate_accuracy(
                predicted=response.content,
                ground_truth=ground_truth,
                alternatives=alternatives,
                use_fuzzy=True,
            )

            # Calculate information-theoretic metrics
            entropy = 0.0
            perplexity = 1.0

            if response.logprobs:
                entropy = entropy_from_logprobs(response.logprobs)
                perplexity = calculate_perplexity(entropy)

            # Calculate composite loss
            loss = calculate_loss(
                entropy=entropy,
                response_length=response.tokens_used,
                perplexity=perplexity,
                accuracy=accuracy,
                normalize=True,
            )

            # Store metrics
            results["metrics"]["accuracies"].append(accuracy)
            results["metrics"]["entropies"].append(entropy)
            results["metrics"]["perplexities"].append(perplexity)
            results["metrics"]["losses"].append(loss)
            results["metrics"]["response_lengths"].append(response.tokens_used)

            # Save individual response if requested
            if save_responses:
                results["responses"].append({
                    "sample_id": sample.get("sample_id"),
                    "question": question,
                    "predicted": response.content,
                    "ground_truth": ground_truth,
                    "correct": accuracy > 0,
                    "metrics": {
                        "accuracy": accuracy,
                        "entropy": entropy,
                        "perplexity": perplexity,
                        "loss": loss,
                        "tokens": response.tokens_used,
                    },
                })

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"      Progress: {i+1}/{len(samples)} samples")

        # Calculate summary statistics
        import numpy as np

        results["summary"] = {
            "mean_accuracy": np.mean(results["metrics"]["accuracies"]),
            "mean_entropy": np.mean(results["metrics"]["entropies"]),
            "mean_perplexity": np.mean(results["metrics"]["perplexities"]),
            "mean_loss": np.mean(results["metrics"]["losses"]),
            "mean_response_length": np.mean(results["metrics"]["response_lengths"]),
            "std_accuracy": np.std(results["metrics"]["accuracies"]),
            "std_entropy": np.std(results["metrics"]["entropies"]),
            "std_perplexity": np.std(results["metrics"]["perplexities"]),
            "std_loss": np.std(results["metrics"]["losses"]),
        }

        print(f"    ✓ Complete: Accuracy={results['summary']['mean_accuracy']:.3f}, "
              f"Loss={results['summary']['mean_loss']:.3f}")

        return results


class PromptOptimizationEvaluator(BaselineEvaluator):
    """
    Evaluates optimized prompt techniques.

    Extends BaselineEvaluator with technique-specific handling.
    """

    def evaluate_with_technique(
        self,
        dataset: Dict[str, Any],
        technique_name: str,
        prompt_template: PromptTemplate,
        save_responses: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate a specific prompt technique.

        Args:
            dataset: Dataset dictionary
            technique_name: Name of technique being evaluated
            prompt_template: Prompt template
            save_responses: Whether to save responses

        Returns:
            Evaluation results
        """
        print(f"    Technique: {technique_name}")

        # Use base evaluator logic
        results = self.evaluate_dataset(dataset, prompt_template, save_responses)

        # Add technique info
        results["technique"] = technique_name
        results["prompt_template"] = {
            "technique": str(prompt_template.technique),
            "has_system_prompt": prompt_template.system_prompt is not None,
            "has_examples": prompt_template.examples is not None,
        }

        return results

    def compare_techniques(
        self,
        dataset: Dict[str, Any],
        techniques: Dict[str, PromptTemplate],
    ) -> Dict[str, Any]:
        """
        Compare multiple techniques on same dataset.

        Args:
            dataset: Dataset to evaluate on
            techniques: Dict mapping technique names to templates

        Returns:
            Comparison results
        """
        comparison = {
            "dataset": dataset.get("dataset_id"),
            "techniques": {},
            "rankings": {},
        }

        # Evaluate each technique
        for tech_name, template in techniques.items():
            results = self.evaluate_with_technique(
                dataset, tech_name, template, save_responses=False
            )
            comparison["techniques"][tech_name] = results["summary"]

        # Rank by accuracy (descending)
        sorted_by_acc = sorted(
            comparison["techniques"].items(),
            key=lambda x: x[1]["mean_accuracy"],
            reverse=True,
        )

        for rank, (tech_name, _) in enumerate(sorted_by_acc, 1):
            comparison["rankings"][tech_name] = rank

        return comparison
