"""
Complete tests for LLM utils to push coverage from 78% to 95%+

Targets llm/utils.py missing lines
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.utils import (
    count_tokens,
    parse_response,
    extract_logprobs,
    normalize_answer,
    fuzzy_match
)


class TestCountTokens:
    """Test token counting with various models."""

    def test_count_tokens_gpt4(self):
        """Test token counting with GPT-4 encoding."""
        text = "This is a test sentence with multiple words."
        count = count_tokens(text, model="gpt-4")

        assert isinstance(count, int)
        assert count > 5  # Should have multiple tokens

    def test_count_tokens_gpt35(self):
        """Test token counting with GPT-3.5 encoding."""
        text = "Hello, world!"
        count = count_tokens(text, model="gpt-3.5-turbo")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_unknown_model(self):
        """Test token counting with unknown model falls back to default."""
        text = "Test text for unknown model"
        # Should use default encoding (cl100k_base)
        count = count_tokens(text, model="unknown-model-name")

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_special_characters(self):
        """Test token counting with special characters."""
        text = "Special chars: 你好 مرحبا שלום"
        count = count_tokens(text)

        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_long_text(self):
        """Test token counting with long text."""
        text = " ".join(["word"] * 500)
        count = count_tokens(text)

        assert count > 400  # Should have many tokens


class TestNormalizeAnswer:
    """Test answer normalization edge cases."""

    def test_normalize_multiple_punctuation(self):
        """Test normalization with multiple punctuation marks."""
        answer = "Hello!!!..."
        normalized = normalize_answer(answer)

        assert normalized == "hello"

    def test_normalize_mixed_case(self):
        """Test normalization with mixed case."""
        answer = "ThIs Is MiXeD cAsE"
        normalized = normalize_answer(answer)

        assert normalized == "this is mixed case"

    def test_normalize_tabs_and_newlines(self):
        """Test normalization with tabs and newlines."""
        answer = "Text\twith\ttabs\nand\nnewlines"
        normalized = normalize_answer(answer)

        assert "\t" not in normalized
        assert "\n" not in normalized
        assert "text" in normalized
        assert "with" in normalized

    def test_normalize_leading_trailing_spaces(self):
        """Test normalization with leading/trailing spaces."""
        answer = "   surrounded by spaces   "
        normalized = normalize_answer(answer)

        assert normalized == "surrounded by spaces"

    def test_normalize_empty_after_punctuation(self):
        """Test normalization of only punctuation."""
        answer = "..."
        normalized = normalize_answer(answer)

        assert normalized == "" or len(normalized) == 0


class TestFuzzyMatch:
    """Test fuzzy matching edge cases."""

    def test_fuzzy_match_partial(self):
        """Test fuzzy match with partial containment."""
        predicted = "The capital of France is Paris"
        ground_truth = "Paris"

        assert fuzzy_match(predicted, ground_truth) is True

    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy match is case insensitive."""
        predicted = "PARIS"
        ground_truth = "paris"

        assert fuzzy_match(predicted, ground_truth) is True

    def test_fuzzy_match_with_punctuation(self):
        """Test fuzzy match handles punctuation."""
        predicted = "Paris."
        ground_truth = "Paris"

        assert fuzzy_match(predicted, ground_truth) is True

    def test_fuzzy_match_alternatives_first(self):
        """Test matching with first alternative."""
        predicted = "four"
        ground_truth = "4"
        alternatives = ["four", "4", "IV"]

        assert fuzzy_match(predicted, ground_truth, alternatives) is True

    def test_fuzzy_match_alternatives_last(self):
        """Test matching with last alternative."""
        predicted = "IV"
        ground_truth = "4"
        alternatives = ["four", "4", "IV"]

        assert fuzzy_match(predicted, ground_truth, alternatives) is True

    def test_fuzzy_match_alternatives_contained(self):
        """Test matching alternative contained in prediction."""
        predicted = "The answer is four"
        ground_truth = "4"
        alternatives = ["four", "4"]

        assert fuzzy_match(predicted, ground_truth, alternatives) is True

    def test_fuzzy_match_no_match_with_alternatives(self):
        """Test no match even with alternatives."""
        predicted = "London"
        ground_truth = "Paris"
        alternatives = ["France capital", "Париж"]

        assert fuzzy_match(predicted, ground_truth, alternatives) is False

    def test_fuzzy_match_empty_prediction(self):
        """Test fuzzy match with empty prediction."""
        predicted = ""
        ground_truth = "Paris"

        assert fuzzy_match(predicted, ground_truth) is False

    def test_fuzzy_match_empty_ground_truth(self):
        """Test fuzzy match with empty ground truth."""
        predicted = "Paris"
        ground_truth = ""

        # Empty ground truth contained in any string
        result = fuzzy_match(predicted, ground_truth)
        assert isinstance(result, bool)

    def test_fuzzy_match_both_empty(self):
        """Test fuzzy match with both empty."""
        assert fuzzy_match("", "") is True


class TestParseResponse:
    """Test response parsing edge cases."""

    def test_parse_response_string_input(self):
        """Test parsing when response is already a string."""
        response = "Simple string response"
        parsed = parse_response(response, provider="openai")

        assert "Simple string" in parsed

    def test_parse_response_unknown_provider(self):
        """Test parsing with unknown provider raises error."""
        try:
            parse_response("test", provider="unknown_provider")
            # Should raise ValueError
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown provider" in str(e)

    def test_parse_response_ollama_dict(self):
        """Test parsing Ollama dict response."""
        response = {"stdout": "  Ollama response  "}
        parsed = parse_response(response, provider="ollama")

        assert parsed == "Ollama response"

    def test_parse_response_ollama_missing_stdout(self):
        """Test parsing Ollama response without stdout."""
        response = {"other_key": "value"}
        parsed = parse_response(response, provider="ollama")

        # Should convert to string
        assert isinstance(parsed, str)


class TestExtractLogprobs:
    """Test logprobs extraction edge cases."""

    def test_extract_logprobs_anthropic_none(self):
        """Test Anthropic logprobs returns None."""
        response = {"content": "test"}
        logprobs = extract_logprobs(response, provider="anthropic")

        assert logprobs is None

    def test_extract_logprobs_ollama_none(self):
        """Test Ollama logprobs returns None."""
        response = {"stdout": "test"}
        logprobs = extract_logprobs(response, provider="ollama")

        assert logprobs is None

    def test_extract_logprobs_unknown_provider(self):
        """Test unknown provider raises error."""
        try:
            extract_logprobs({}, provider="unknown")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Unknown provider" in str(e)

    def test_extract_logprobs_openai_no_logprobs(self):
        """Test OpenAI response without logprobs."""
        class MockChoice:
            def __init__(self):
                self.logprobs = None

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        response = MockResponse()
        logprobs = extract_logprobs(response, provider="openai")

        assert logprobs is None

    def test_extract_logprobs_openai_empty_content(self):
        """Test OpenAI logprobs with empty content."""
        class MockLogprobs:
            def __init__(self):
                self.content = []

        class MockChoice:
            def __init__(self):
                self.logprobs = MockLogprobs()

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        response = MockResponse()
        logprobs = extract_logprobs(response, provider="openai")

        assert logprobs is None
