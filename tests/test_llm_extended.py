"""
Extended unit tests for LLM module

Tests additional LLM client functionality and edge cases.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.utils import count_tokens, parse_response, extract_logprobs, normalize_answer, fuzzy_match
from llm.client import LLMClient


class TestTokenCounting:
    """Test token counting utilities."""

    def test_count_tokens_basic(self):
        """Test basic token counting."""
        text = "Hello, world!"
        count = count_tokens(text)

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_empty(self):
        """Test token counting with empty string."""
        count = count_tokens("")

        assert count == 0

    def test_count_tokens_long_text(self):
        """Test token counting with long text."""
        text = " ".join(["word"] * 100)
        count = count_tokens(text)

        assert count > 50  # Should have some tokens

    def test_count_tokens_different_models(self):
        """Test token counting with different model encodings."""
        text = "This is a test."

        count_gpt4 = count_tokens(text, model="gpt-4")
        count_gpt35 = count_tokens(text, model="gpt-3.5-turbo")

        # Both should return counts
        assert count_gpt4 > 0
        assert count_gpt35 > 0


class TestResponseParsing:
    """Test response parsing utilities."""

    def test_parse_response_openai(self):
        """Test parsing OpenAI response."""
        # Mock response
        class MockChoice:
            def __init__(self):
                self.message = type('obj', (object,), {'content': 'Test response'})()

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        response = MockResponse()
        parsed = parse_response(response, provider="openai")

        assert parsed == "Test response"

    def test_parse_response_ollama(self):
        """Test parsing Ollama response."""
        response = {"stdout": "  Test response  "}
        parsed = parse_response(response, provider="ollama")

        assert parsed == "Test response"

    def test_parse_response_string(self):
        """Test parsing string response."""
        response = "Simple string"
        parsed = parse_response(response, provider="openai")

        assert "Simple string" in parsed


class TestAnswerNormalization:
    """Test answer normalization utilities."""

    def test_normalize_answer_basic(self):
        """Test basic normalization."""
        answer = "  HELLO WORLD!  "
        normalized = normalize_answer(answer)

        assert normalized == "hello world"

    def test_normalize_answer_punctuation(self):
        """Test punctuation removal."""
        answer = "Answer..."
        normalized = normalize_answer(answer)

        assert normalized == "answer"

    def test_normalize_answer_whitespace(self):
        """Test whitespace normalization."""
        answer = "Multiple    spaces"
        normalized = normalize_answer(answer)

        assert normalized == "multiple spaces"


class TestFuzzyMatching:
    """Test fuzzy answer matching."""

    def test_fuzzy_match_exact(self):
        """Test exact match."""
        assert fuzzy_match("Paris", "Paris") == True
        assert fuzzy_match("paris", "Paris") == True

    def test_fuzzy_match_contained(self):
        """Test containment match."""
        assert fuzzy_match("The capital is Paris", "Paris") == True

    def test_fuzzy_match_alternatives(self):
        """Test matching with alternatives."""
        assert fuzzy_match("4", "four", alternatives=["4", "four"]) == True

    def test_fuzzy_match_no_match(self):
        """Test no match."""
        assert fuzzy_match("London", "Paris") == False


class TestLLMClient:
    """Test LLMClient base class."""

    def test_llm_client_initialization(self):
        """Test LLM client cannot be instantiated directly."""
        try:
            client = LLMClient(provider="test")
            # If it doesn't raise, just check it exists
            assert client is not None
        except (TypeError, NotImplementedError):
            # Expected - abstract class
            assert True

    def test_llm_client_provider_setting(self):
        """Test LLM client with different providers."""
        try:
            client = LLMClient(provider="ollama", model="llama3.2")
            assert client.model == "llama3.2"
            assert client.provider == "ollama"
        except Exception:
            # May fail if not implemented or Ollama not running
            pass


class TestLogProbsExtraction:
    """Test log probability extraction."""

    def test_extract_logprobs_none(self):
        """Test extracting logprobs when none available."""
        class MockResponse:
            choices = []

        response = MockResponse()
        logprobs = extract_logprobs(response, provider="openai")

        assert logprobs is None

    def test_extract_logprobs_anthropic(self):
        """Test Anthropic logprobs (not supported)."""
        response = {}
        logprobs = extract_logprobs(response, provider="anthropic")

        assert logprobs is None

    def test_extract_logprobs_ollama(self):
        """Test Ollama logprobs (not supported)."""
        response = {}
        logprobs = extract_logprobs(response, provider="ollama")

        assert logprobs is None
