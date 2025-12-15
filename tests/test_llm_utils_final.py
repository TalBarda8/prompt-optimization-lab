"""
Final tests to cover remaining llm/utils.py lines (55, 88-89)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.utils import parse_response, extract_logprobs


class TestParseResponseAnthropicFallback:
    """Cover line 55 - Anthropic response fallback to str()."""

    def test_anthropic_response_without_content(self):
        """Test Anthropic response that lacks content attribute."""
        # Mock response without proper content structure
        class MockAnthropicResponse:
            def __init__(self):
                pass  # No content attribute

        response = MockAnthropicResponse()
        parsed = parse_response(response, provider="anthropic")

        # Should fall back to str(response) - line 55
        assert isinstance(parsed, str)

    def test_anthropic_response_with_empty_content(self):
        """Test Anthropic response with empty content list."""
        class MockAnthropicResponse:
            def __init__(self):
                self.content = []  # Empty content

        response = MockAnthropicResponse()
        parsed = parse_response(response, provider="anthropic")

        # Should fall back to str(response) - line 55
        assert isinstance(parsed, str)


class TestExtractLogprobsWithContent:
    """Cover lines 88-89 - OpenAI logprobs extraction loop."""

    def test_extract_openai_logprobs_with_content(self):
        """Test extracting actual logprobs from OpenAI response."""
        # Mock OpenAI response with logprobs
        class MockTokenLogprob:
            def __init__(self, token, logprob):
                self.token = token
                self.logprob = logprob
                self.bytes = None

        class MockLogprobs:
            def __init__(self):
                self.content = [
                    MockTokenLogprob("The", -0.1),
                    MockTokenLogprob("quick", -0.5),
                ]

        class MockChoice:
            def __init__(self):
                self.logprobs = MockLogprobs()

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        response = MockResponse()
        logprobs = extract_logprobs(response, provider="openai")

        # Should extract logprobs via loop on lines 88-89
        assert logprobs is not None
        assert len(logprobs) == 2
        assert logprobs[0]['token'] == "The"
        assert logprobs[0]['logprob'] == -0.1
        assert logprobs[1]['token'] == "quick"

    def test_extract_openai_logprobs_multiple_tokens(self):
        """Test extracting logprobs with many tokens."""
        class MockTokenLogprob:
            def __init__(self, token, logprob):
                self.token = token
                self.logprob = logprob

        class MockLogprobs:
            def __init__(self):
                # Multiple tokens to ensure loop runs multiple times
                self.content = [
                    MockTokenLogprob(f"token{i}", -i * 0.1)
                    for i in range(10)
                ]

        class MockChoice:
            def __init__(self):
                self.logprobs = MockLogprobs()

        class MockResponse:
            def __init__(self):
                self.choices = [MockChoice()]

        response = MockResponse()
        logprobs = extract_logprobs(response, provider="openai")

        # Should extract all 10 tokens via loop
        assert logprobs is not None
        assert len(logprobs) == 10
        for i in range(10):
            assert logprobs[i]['token'] == f"token{i}"
            assert logprobs[i]['logprob'] == -i * 0.1
