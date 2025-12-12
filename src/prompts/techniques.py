"""
Prompt Engineering Techniques

Implements 6+ techniques from PRD Section 3:
1. Baseline (direct questioning)
2. Chain-of-Thought (CoT)
3. Chain-of-Thought++ (CoT++)
4. ReAct (Reasoning + Acting)
5. Tree-of-Thoughts (ToT)
6. Role-Based Prompting
7. Few-Shot Learning
"""

from typing import List, Dict, Any, Optional
from .base import PromptTemplate, PromptTechnique, BasePromptBuilder


class BaselinePrompt(BasePromptBuilder):
    """
    Baseline: Direct questioning without guidance.

    PRD Section 3.1: Control group for comparison.
    """

    def __init__(self):
        super().__init__(PromptTechnique.BASELINE)

    def build(self, **kwargs) -> PromptTemplate:
        """Build baseline prompt."""
        return PromptTemplate(
            technique=self.technique,
            system_prompt=None,
            user_prompt="{question}",
            metadata={"description": "Baseline direct questioning"},
        )


class ChainOfThoughtPrompt(BasePromptBuilder):
    """
    Chain-of-Thought (CoT): Encourage step-by-step reasoning.

    PRD Section 3.2: "Let's think step by step"
    Promotes explicit reasoning chains.
    """

    def __init__(self):
        super().__init__(PromptTechnique.CHAIN_OF_THOUGHT)

    def build(self, **kwargs) -> PromptTemplate:
        """Build CoT prompt."""
        system_prompt = (
            "You are a helpful assistant that thinks step-by-step. "
            "When solving problems, break down your reasoning into clear steps."
        )

        user_prompt = (
            "{question}\n\n"
            "Let's approach this step-by-step:\n"
            "1. First, identify what we need to find\n"
            "2. Then, work through the problem systematically\n"
            "3. Finally, state the answer clearly"
        )

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "description": "Chain-of-Thought prompting",
                "instruction": "Let's think step by step",
            },
        )


class ChainOfThoughtPlusPlusPrompt(BasePromptBuilder):
    """
    Chain-of-Thought++ (CoT++): Enhanced CoT with verification.

    PRD Section 3.3: CoT + self-verification + confidence scoring.
    """

    def __init__(self):
        super().__init__(PromptTechnique.CHAIN_OF_THOUGHT_PLUS_PLUS)

    def build(self, **kwargs) -> PromptTemplate:
        """Build CoT++ prompt."""
        system_prompt = (
            "You are an expert problem solver that uses rigorous step-by-step reasoning. "
            "Always verify your work and indicate your confidence level."
        )

        user_prompt = (
            "{question}\n\n"
            "Solve this problem using the following structure:\n"
            "1. **Understanding**: Restate what the problem is asking\n"
            "2. **Reasoning**: Work through the problem step-by-step\n"
            "3. **Verification**: Check your work for errors\n"
            "4. **Confidence**: Rate your confidence (Low/Medium/High)\n"
            "5. **Answer**: State the final answer clearly"
        )

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "description": "CoT++ with verification and confidence",
                "features": ["self-verification", "confidence_scoring"],
            },
        )


class ReActPrompt(BasePromptBuilder):
    """
    ReAct: Reasoning + Acting in interleaved manner.

    PRD Section 3.4: Alternates between reasoning and action steps.
    Useful for problems requiring intermediate decisions.
    """

    def __init__(self):
        super().__init__(PromptTechnique.REACT)

    def build(self, **kwargs) -> PromptTemplate:
        """Build ReAct prompt."""
        system_prompt = (
            "You are a systematic problem solver that alternates between "
            "thinking (reasoning) and doing (taking action steps)."
        )

        user_prompt = (
            "{question}\n\n"
            "Use the ReAct framework:\n"
            "- **Thought**: What do I need to consider?\n"
            "- **Action**: What step should I take?\n"
            "- **Observation**: What did I learn?\n"
            "Repeat this cycle until you reach the answer.\n\n"
            "Format:\n"
            "Thought 1: [your reasoning]\n"
            "Action 1: [step taken]\n"
            "Observation 1: [result]\n"
            "...\n"
            "Final Answer: [conclusion]"
        )

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "description": "ReAct reasoning and acting",
                "pattern": "thought-action-observation",
            },
        )


class TreeOfThoughtsPrompt(BasePromptBuilder):
    """
    Tree-of-Thoughts (ToT): Explore multiple reasoning paths.

    PRD Section 3.5: Consider alternative approaches before committing.
    """

    def __init__(self):
        super().__init__(PromptTechnique.TREE_OF_THOUGHTS)

    def build(self, **kwargs) -> PromptTemplate:
        """Build ToT prompt."""
        system_prompt = (
            "You are an expert problem solver that explores multiple solution paths "
            "before deciding on the best approach."
        )

        user_prompt = (
            "{question}\n\n"
            "Use the Tree-of-Thoughts approach:\n"
            "1. **Identify Approaches**: List 2-3 different ways to solve this\n"
            "2. **Evaluate Approaches**: Assess pros/cons of each\n"
            "3. **Select Best Path**: Choose the most promising approach\n"
            "4. **Execute**: Solve using the selected approach\n"
            "5. **Verify**: Confirm the answer is correct"
        )

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "description": "Tree-of-Thoughts multi-path exploration",
                "features": ["multiple_approaches", "path_selection"],
            },
        )


class RoleBasedPrompt(BasePromptBuilder):
    """
    Role-Based Prompting: Assign expert role/persona.

    PRD Section 3.6: Leverage role-playing for domain expertise.
    """

    def __init__(self, role: str = "expert"):
        """
        Initialize role-based prompt.

        Args:
            role: Role to assign (e.g., "expert", "teacher", "scientist")
        """
        super().__init__(PromptTechnique.ROLE_BASED)
        self.role = role

    def build(self, role: Optional[str] = None, **kwargs) -> PromptTemplate:
        """Build role-based prompt."""
        role = role or self.role

        # Role-specific system prompts
        role_prompts = {
            "expert": (
                "You are a world-class expert with deep knowledge across multiple domains. "
                "Apply your expertise to solve problems accurately and thoroughly."
            ),
            "teacher": (
                "You are an experienced teacher who excels at explaining complex concepts clearly. "
                "Break down problems into understandable steps for learners."
            ),
            "scientist": (
                "You are a rigorous scientist who relies on logic, evidence, and systematic methods. "
                "Approach problems with scientific precision and verify all conclusions."
            ),
            "mathematician": (
                "You are a skilled mathematician with expertise in problem-solving. "
                "Apply mathematical reasoning and verify all calculations carefully."
            ),
        }

        system_prompt = role_prompts.get(
            role,
            f"You are a {role} with specialized expertise. Use your knowledge to solve this problem."
        )

        user_prompt = "{question}"

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata={
                "description": f"Role-based prompting as {role}",
                "role": role,
            },
        )


class FewShotPrompt(BasePromptBuilder):
    """
    Few-Shot Learning: Provide examples before the question.

    PRD Section 3.7: Learn from examples to improve performance.
    """

    def __init__(self, examples: Optional[List[Dict[str, str]]] = None):
        """
        Initialize few-shot prompt.

        Args:
            examples: List of example dicts with 'question' and 'answer' keys
        """
        super().__init__(PromptTechnique.FEW_SHOT)
        self.default_examples = examples or []

    def build(self, examples: Optional[List[Dict[str, str]]] = None, **kwargs) -> PromptTemplate:
        """Build few-shot prompt."""
        examples = examples or self.default_examples

        system_prompt = (
            "You are a helpful assistant. Learn from the examples below to "
            "understand the expected format and reasoning style."
        )

        # Build examples section
        examples_text = ""
        if examples:
            examples_text = "Here are some examples:\n\n"
            for i, ex in enumerate(examples, 1):
                examples_text += f"Example {i}:\n"
                examples_text += f"Q: {ex['question']}\n"
                examples_text += f"A: {ex['answer']}\n\n"

        user_prompt = examples_text + "Now solve this problem:\nQ: {question}\nA:"

        return PromptTemplate(
            technique=self.technique,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            examples=examples,
            metadata={
                "description": "Few-shot learning with examples",
                "n_examples": len(examples),
            },
        )


# Convenience functions for quick access
def get_baseline_prompt() -> PromptTemplate:
    """Get baseline prompt template."""
    return BaselinePrompt().build()


def get_cot_prompt() -> PromptTemplate:
    """Get Chain-of-Thought prompt template."""
    return ChainOfThoughtPrompt().build()


def get_cot_plus_plus_prompt() -> PromptTemplate:
    """Get Chain-of-Thought++ prompt template."""
    return ChainOfThoughtPlusPlusPrompt().build()


def get_react_prompt() -> PromptTemplate:
    """Get ReAct prompt template."""
    return ReActPrompt().build()


def get_tot_prompt() -> PromptTemplate:
    """Get Tree-of-Thoughts prompt template."""
    return TreeOfThoughtsPrompt().build()


def get_role_based_prompt(role: str = "expert") -> PromptTemplate:
    """Get role-based prompt template."""
    return RoleBasedPrompt(role=role).build()


def get_few_shot_prompt(examples: List[Dict[str, str]]) -> PromptTemplate:
    """Get few-shot prompt template."""
    return FewShotPrompt(examples=examples).build()
