"""
Unit tests for LLM module

Tests client initialization, utilities, and response parsing.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm import LLMClient, LLMResponse
from llm.utils import (
    count_tokens,
    normalize_answer,
    fuzzy_match,
    parse_response,
)


class TestLLMUtils:
    """Test LLM utility functions."""

    def test_count_tokens_simple(self):
        """Test token counting."""
        text = "Hello, world!"
        tokens = count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_count_tokens_empty(self):
        """Test token counting with empty string."""
        tokens = count_tokens("")
        assert tokens == 0

    def test_normalize_answer(self):
        """Test answer normalization."""
        # Lowercase
        assert normalize_answer("PARIS") == "paris"

        # Trim whitespace
        assert normalize_answer("  Paris  ") == "paris"

        # Remove extra whitespace
        assert normalize_answer("New   York") == "new york"

        # Remove trailing punctuation
        assert normalize_answer("Paris.") == "paris"
        assert normalize_answer("42!") == "42"

    def test_fuzzy_match_exact(self):
        """Test exact fuzzy match."""
        assert fuzzy_match("Paris", "Paris") is True
        assert fuzzy_match("paris", "Paris") is True
        assert fuzzy_match("  Paris  ", "Paris") is True

    def test_fuzzy_match_contained(self):
        """Test contained fuzzy match."""
        assert fuzzy_match("The capital is Paris", "Paris") is True
        assert fuzzy_match("Paris, France", "Paris") is True

    def test_fuzzy_match_alternatives(self):
        """Test fuzzy match with alternatives."""
        assert fuzzy_match("US", "United States", ["US", "USA", "America"]) is True
        assert fuzzy_match("USA", "United States", ["US", "USA", "America"]) is True

    def test_fuzzy_match_no_match(self):
        """Test fuzzy match with no match."""
        assert fuzzy_match("London", "Paris") is False
        assert fuzzy_match("42", "43") is False

    def test_parse_response_openai(self):
        """Test parsing OpenAI response."""
        # Mock OpenAI response
        mock_message = Mock()
        mock_message.content = "This is a test response"

        mock_choice = Mock()
        mock_choice.message = mock_message

        mock_response = Mock()
        mock_response.choices = [mock_choice]

        content = parse_response(mock_response, provider="openai")
        assert content == "This is a test response"

    def test_parse_response_anthropic(self):
        """Test parsing Anthropic response."""
        # Mock Anthropic response
        mock_content = Mock()
        mock_content.text = "This is a test response"

        mock_response = Mock()
        mock_response.content = [mock_content]

        content = parse_response(mock_response, provider="anthropic")
        assert content == "This is a test response"


class TestLLMClient:
    """Test LLM client initialization and configuration."""

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('llm.client.OpenAI')
    def test_init_openai(self, mock_openai_class):
        """Test OpenAI client initialization."""
        client = LLMClient(provider="openai", model="gpt-4")

        assert client.provider == "openai"
        assert client.model == "gpt-4"
        assert client.temperature == 0.0
        assert client.max_tokens == 500

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'})
    @patch('llm.client.Anthropic')
    def test_init_anthropic(self, mock_anthropic_class):
        """Test Anthropic client initialization."""
        client = LLMClient(provider="anthropic", model="claude-3-opus-20240229")

        assert client.provider == "anthropic"
        assert client.model == "claude-3-opus-20240229"

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('llm.client.OpenAI')
    def test_client_custom_params(self, mock_openai_class):
        """Test client with custom parameters."""
        client = LLMClient(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1000,
        )

        assert client.temperature == 0.7
        assert client.max_tokens == 1000

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('llm.client.OpenAI')
    def test_count_prompt_tokens(self, mock_openai_class):
        """Test prompt token counting."""
        client = LLMClient(provider="openai")

        prompt = "What is the capital of France?"
        tokens = client.count_prompt_tokens(prompt)

        assert tokens > 0
        assert isinstance(tokens, int)

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    @patch('llm.client.OpenAI')
    def test_count_prompt_tokens_with_system(self, mock_openai_class):
        """Test token counting with system prompt."""
        client = LLMClient(provider="openai")

        prompt = "What is the capital of France?"
        system_prompt = "You are a helpful assistant."
        tokens = client.count_prompt_tokens(prompt, system_prompt)

        # Should be more tokens than prompt alone
        prompt_only_tokens = client.count_prompt_tokens(prompt)
        assert tokens > prompt_only_tokens


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Test response",
            tokens_used=100,
            model="gpt-4",
            provider="openai",
        )

        assert response.content == "Test response"
        assert response.tokens_used == 100
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.logprobs is None

    def test_response_with_logprobs(self):
        """Test LLMResponse with logprobs."""
        logprobs = [
            {"token": "Hello", "logprob": -0.5},
            {"token": " world", "logprob": -1.2},
        ]

        response = LLMResponse(
            content="Hello world",
            logprobs=logprobs,
            tokens_used=50,
            model="gpt-4",
            provider="openai",
        )

        assert response.logprobs is not None
        assert len(response.logprobs) == 2
        assert response.logprobs[0]["token"] == "Hello"


# Run tests if executed directly
if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
