"""
Unit tests for prompts module

Tests all 6+ prompt engineering techniques.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompts import (
    PromptTemplate,
    PromptTechnique,
    BaselinePrompt,
    ChainOfThoughtPrompt,
    ChainOfThoughtPlusPlusPrompt,
    ReActPrompt,
    TreeOfThoughtsPrompt,
    RoleBasedPrompt,
    FewShotPrompt,
)


class TestPromptTemplate:
    """Test base PromptTemplate class."""

    def test_template_creation(self):
        """Test creating a prompt template."""
        template = PromptTemplate(
            technique=PromptTechnique.BASELINE,
            user_prompt="What is {question}?",
        )

        assert template.technique == PromptTechnique.BASELINE
        assert "{question}" in template.user_prompt

    def test_template_format(self):
        """Test formatting a template with a question."""
        template = PromptTemplate(
            technique=PromptTechnique.BASELINE,
            user_prompt="What is {question}?",
        )

        formatted = template.format(question="the capital of France")
        assert formatted == "What is the capital of France?"

    def test_get_full_prompt(self):
        """Test getting full prompt with system and user parts."""
        template = PromptTemplate(
            technique=PromptTechnique.BASELINE,
            system_prompt="You are helpful.",
            user_prompt="Q: {question}",
        )

        full = template.get_full_prompt(question="What is 2+2?")
        assert full["system"] == "You are helpful."
        assert "What is 2+2?" in full["user"]


class TestBaselinePrompt:
    """Test baseline prompt technique."""

    def test_baseline_build(self):
        """Test building baseline prompt."""
        builder = BaselinePrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.BASELINE
        assert template.system_prompt is None
        assert "{question}" in template.user_prompt

    def test_baseline_format(self):
        """Test baseline prompt formatting."""
        builder = BaselinePrompt()
        template = builder.build()

        formatted = template.format(question="What is the capital of France?")
        assert "What is the capital of France?" in formatted


class TestChainOfThoughtPrompt:
    """Test Chain-of-Thought prompt."""

    def test_cot_build(self):
        """Test building CoT prompt."""
        builder = ChainOfThoughtPrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.CHAIN_OF_THOUGHT
        assert template.system_prompt is not None
        assert "step" in template.user_prompt.lower()

    def test_cot_has_structure(self):
        """Test that CoT includes step-by-step structure."""
        builder = ChainOfThoughtPrompt()
        template = builder.build()

        formatted = template.format(question="Calculate 2+2")
        assert "step" in formatted.lower()


class TestChainOfThoughtPlusPlusPrompt:
    """Test Chain-of-Thought++ prompt."""

    def test_cot_plus_plus_build(self):
        """Test building CoT++ prompt."""
        builder = ChainOfThoughtPlusPlusPrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.CHAIN_OF_THOUGHT_PLUS_PLUS
        assert "verification" in template.user_prompt.lower() or "verify" in template.user_prompt.lower()
        assert "confidence" in template.user_prompt.lower()

    def test_cot_plus_plus_has_verification(self):
        """Test that CoT++ includes verification step."""
        builder = ChainOfThoughtPlusPlusPrompt()
        template = builder.build()

        assert "verif" in template.user_prompt.lower()


class TestReActPrompt:
    """Test ReAct prompt."""

    def test_react_build(self):
        """Test building ReAct prompt."""
        builder = ReActPrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.REACT
        assert "thought" in template.user_prompt.lower()
        assert "action" in template.user_prompt.lower()

    def test_react_has_cycle(self):
        """Test that ReAct includes thought-action-observation cycle."""
        builder = ReActPrompt()
        template = builder.build()

        prompt_lower = template.user_prompt.lower()
        assert "thought" in prompt_lower
        assert "action" in prompt_lower
        assert "observation" in prompt_lower


class TestTreeOfThoughtsPrompt:
    """Test Tree-of-Thoughts prompt."""

    def test_tot_build(self):
        """Test building ToT prompt."""
        builder = TreeOfThoughtsPrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.TREE_OF_THOUGHTS
        assert "approach" in template.user_prompt.lower()

    def test_tot_has_multiple_paths(self):
        """Test that ToT considers multiple approaches."""
        builder = TreeOfThoughtsPrompt()
        template = builder.build()

        prompt_lower = template.user_prompt.lower()
        assert "approach" in prompt_lower or "path" in prompt_lower


class TestRoleBasedPrompt:
    """Test role-based prompt."""

    def test_role_based_build_default(self):
        """Test building role-based prompt with default role."""
        builder = RoleBasedPrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.ROLE_BASED
        assert template.system_prompt is not None
        assert "expert" in template.system_prompt.lower()

    def test_role_based_build_teacher(self):
        """Test building role-based prompt with teacher role."""
        builder = RoleBasedPrompt(role="teacher")
        template = builder.build()

        assert "teacher" in template.system_prompt.lower()

    def test_role_based_build_scientist(self):
        """Test building role-based prompt with scientist role."""
        builder = RoleBasedPrompt(role="scientist")
        template = builder.build()

        assert "scientist" in template.system_prompt.lower()

    def test_role_based_custom_role(self):
        """Test building role-based prompt with custom role."""
        builder = RoleBasedPrompt()
        template = builder.build(role="detective")

        assert "detective" in template.system_prompt.lower()


class TestFewShotPrompt:
    """Test few-shot learning prompt."""

    def test_few_shot_build_no_examples(self):
        """Test building few-shot prompt without examples."""
        builder = FewShotPrompt()
        template = builder.build()

        assert template.technique == PromptTechnique.FEW_SHOT
        assert template.examples is not None

    def test_few_shot_build_with_examples(self):
        """Test building few-shot prompt with examples."""
        examples = [
            {"question": "What is 2+2?", "answer": "4"},
            {"question": "What is 3+3?", "answer": "6"},
        ]

        builder = FewShotPrompt(examples=examples)
        template = builder.build()

        assert len(template.examples) == 2
        assert "2+2" in template.user_prompt
        assert "3+3" in template.user_prompt

    def test_few_shot_format_includes_examples(self):
        """Test that formatted few-shot prompt includes examples."""
        examples = [
            {"question": "What is 2+2?", "answer": "4"},
        ]

        builder = FewShotPrompt(examples=examples)
        template = builder.build()

        formatted = template.format(question="What is 5+5?")
        assert "2+2" in formatted  # Example included
        assert "5+5" in formatted  # New question included


class TestPromptBuilderValidation:
    """Test prompt builder validation."""

    def test_baseline_builder_validates(self):
        """Test that builder validates its own templates."""
        builder = BaselinePrompt()
        template = builder.build()

        assert builder.validate(template) is True

    def test_builder_rejects_wrong_technique(self):
        """Test that builder rejects templates with wrong technique."""
        builder = BaselinePrompt()
        template = PromptTemplate(
            technique=PromptTechnique.CHAIN_OF_THOUGHT,  # Wrong technique
            user_prompt="Test",
        )

        assert builder.validate(template) is False

    def test_builder_rejects_empty_prompt(self):
        """Test that builder rejects templates with empty prompt."""
        builder = BaselinePrompt()
        template = PromptTemplate(
            technique=PromptTechnique.BASELINE,
            user_prompt="",  # Empty
        )

        assert builder.validate(template) is False


# Run tests if executed directly
if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
