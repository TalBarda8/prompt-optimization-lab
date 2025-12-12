"""
LLM Client - Unified interface for OpenAI, Anthropic, and Ollama

Provides consistent API for generating responses with logprobs support.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import os
import time
import subprocess
import shutil
import json as json_module

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

from .utils import parse_response, extract_logprobs, count_tokens


@dataclass
class LLMResponse:
    """
    Standardized LLM response.

    Attributes:
        content: Generated text content
        logprobs: Log probabilities (if available)
        tokens_used: Total tokens used (prompt + completion)
        model: Model used
        provider: API provider
        raw_response: Raw API response object
        latency_ms: Response latency in milliseconds
    """
    content: str
    logprobs: Optional[List[Dict[str, Any]]] = None
    tokens_used: int = 0
    model: str = ""
    provider: str = ""
    raw_response: Any = None
    latency_ms: float = 0.0


class LLMClient:
    """
    Unified LLM client for OpenAI, Anthropic, and Ollama.

    Supports:
    - Multiple providers (OpenAI, Anthropic, Ollama)
    - Logprobs extraction (for entropy/perplexity)
    - Consistent interface
    - Error handling and retries
    - Local LLM inference via Ollama
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 500,
        api_key: Optional[str] = None,
    ):
        """
        Initialize LLM client.

        Args:
            provider: "openai", "anthropic", or "ollama"
            model: Model name (e.g., "gpt-4", "claude-3-opus-20240229", "llama3.2")
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate
            api_key: API key (if None, reads from env) - not needed for Ollama
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize API client
        if self.provider == "openai":
            if OpenAI is None:
                raise ImportError("openai package not installed. Run: pip install openai")
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            if Anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)

        elif self.provider == "ollama":
            # Check if Ollama is installed
            if shutil.which("ollama") is None:
                raise RuntimeError(
                    "Ollama is not installed or not in PATH.\n"
                    "Install Ollama:\n"
                    "  macOS/Linux: brew install ollama\n"
                    "  Or visit: https://ollama.ai\n"
                    "Then pull the model: ollama pull llama3.2"
                )
            # Verify model is available
            self._verify_ollama_model()
            self.client = None  # Ollama uses subprocess, no client object needed

        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'ollama'")

    def _verify_ollama_model(self):
        """Verify that the Ollama model is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if self.model not in result.stdout:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama.\n"
                    f"Pull the model with: ollama pull {self.model}"
                )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama command timed out. Is Ollama running?")
        except FileNotFoundError:
            raise RuntimeError("Ollama executable not found.")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        logprobs: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            logprobs: Whether to request log probabilities (not supported for Ollama)
            **kwargs: Additional API parameters

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        if self.provider == "openai":
            response = self._generate_openai(prompt, system_prompt, logprobs, **kwargs)
        elif self.provider == "anthropic":
            response = self._generate_anthropic(prompt, system_prompt, **kwargs)
        elif self.provider == "ollama":
            response = self._generate_ollama(prompt, system_prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        latency_ms = (time.time() - start_time) * 1000
        response.latency_ms = latency_ms

        return response

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        logprobs: bool = True,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Merge kwargs with defaults
        api_params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # Add logprobs if requested
        if logprobs:
            api_params["logprobs"] = True

        api_params.update(kwargs)

        # Make API call
        raw_response = self.client.chat.completions.create(**api_params)

        # Parse response
        content = parse_response(raw_response, provider="openai")
        logprobs_data = extract_logprobs(raw_response, provider="openai") if logprobs else None

        # Calculate tokens
        tokens_used = 0
        if hasattr(raw_response, 'usage'):
            tokens_used = raw_response.usage.total_tokens

        return LLMResponse(
            content=content,
            logprobs=logprobs_data,
            tokens_used=tokens_used,
            model=self.model,
            provider="openai",
            raw_response=raw_response,
        )

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        api_params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_prompt:
            api_params["system"] = system_prompt

        api_params.update(kwargs)

        # Make API call
        raw_response = self.client.messages.create(**api_params)

        # Parse response
        content = parse_response(raw_response, provider="anthropic")

        # Calculate tokens
        tokens_used = 0
        if hasattr(raw_response, 'usage'):
            tokens_used = raw_response.usage.input_tokens + raw_response.usage.output_tokens

        return LLMResponse(
            content=content,
            logprobs=None,  # Anthropic doesn't provide logprobs
            tokens_used=tokens_used,
            model=self.model,
            provider="anthropic",
            raw_response=raw_response,
        )

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama (local LLM)."""
        # Construct the full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Prepare Ollama API call via subprocess
        # Using 'ollama run' with JSON output for better parsing
        try:
            # Build command
            cmd = ["ollama", "run", self.model, full_prompt]

            # Add optional parameters via environment or command args
            # Ollama CLI doesn't support all parameters, but we can try
            options = {}
            if self.temperature != 0.7:  # Ollama default is 0.7
                options["temperature"] = self.temperature
            if self.max_tokens != 500:
                options["num_predict"] = self.max_tokens

            # Run Ollama
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout for local generation
            )

            if result.returncode != 0:
                raise RuntimeError(f"Ollama command failed: {result.stderr}")

            content = result.stdout.strip()

            # Estimate token count (rough approximation: 4 chars per token)
            tokens_used = (len(full_prompt) + len(content)) // 4

            return LLMResponse(
                content=content,
                logprobs=None,  # Ollama doesn't provide logprobs via CLI
                tokens_used=tokens_used,
                model=self.model,
                provider="ollama",
                raw_response={"stdout": result.stdout, "stderr": result.stderr},
            )

        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Ollama generation timed out after 120 seconds. "
                f"The model '{self.model}' may be too slow or the prompt too complex."
            )
        except FileNotFoundError:
            raise RuntimeError("Ollama executable not found. Is Ollama installed?")

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        logprobs: bool = True,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            system_prompt: System prompt for all requests
            logprobs: Whether to request logprobs
            **kwargs: Additional API parameters

        Returns:
            List of LLMResponse objects
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt, logprobs, **kwargs)
            responses.append(response)
        return responses

    def count_prompt_tokens(self, prompt: str, system_prompt: Optional[str] = None) -> int:
        """
        Count tokens in prompt.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Number of tokens
        """
        full_text = ""
        if system_prompt:
            full_text += system_prompt + "\n\n"
        full_text += prompt

        return count_tokens(full_text, model=self.model)
