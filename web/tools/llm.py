import os
import json 
import time
import requests
from typing import Dict, Any, Optional

from ..providers.llm_http import HTTPLLMProvider


class SafetyLLM:
    """LLM Provider Router for multiple backends (openrouter|http|local)"""

    def __init__(
        self,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the safety LLM router

        Args:
            model: Model identifier (defaults to ANALYSIS_LLM_MODEL env var)
            backend: Backend type (defaults to ANALYSIS_LLM_BACKEND env var)
            api_key: API key for cloud providers
            **kwargs: Additional provider-specific arguments
        """
        self.backend = backend or os.getenv("ANALYSIS_LLM_BACKEND", "openrouter")
        self.model = model or kwargs.get("model_name") or os.getenv("ANALYSIS_LLM_MODEL", "SafeLens/llama-3-8b")
        self.timeout = int(os.getenv("ANALYSIS_LLM_TIMEOUT_SEC", "30"))

        if self.backend == "openrouter":
            self.provider = self._init_openrouter_provider(api_key)
        elif self.backend == "http":
            self.provider = self._init_http_provider()
        elif self.backend == "local":
            self.provider = self._init_local_provider()
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend}. "
                f"Supported backends: openrouter, http, local"
            )

    def _init_openrouter_provider(self, api_key: Optional[str]):
        """Initialize OpenRouter or direct OpenAI-compatible provider (like Google AI Studio)"""
        base_url = os.getenv("ANALYSIS_LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.api_url = f"{base_url}/chat/completions"
        self.api_key = api_key or os.getenv("ANALYSIS_LLM_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            return MockLLMProvider()

        return OpenRouterProvider(self.api_url, self.model, self.api_key, self.timeout)

    def _init_http_provider(self):
        """Initialize HTTP provider"""
        base_url = os.getenv("ANALYSIS_LLM_HTTP_URL")
        if not base_url:
            raise ValueError("ANALYSIS_LLM_HTTP_URL is required for HTTP backend")

        headers = {}
        headers_str = os.getenv("ANALYSIS_LLM_HTTP_HEADERS")
        if headers_str:
            try:
                headers = json.loads(headers_str)
            except json.JSONDecodeError:
                raise ValueError("ANALYSIS_LLM_HTTP_HEADERS must be valid JSON")

        return HTTPLLMProvider(base_url, self.model, headers, self.timeout)

    def _init_local_provider(self):
        """Initialize local provider.

        Tries to import an optional LocalLLMProvider from web/providers/llm_local.py.
        If it is missing, fall back to a lightweight mock so the server can run
        without local LLM dependencies.
        """
        try:
            from ..providers.llm_local import LocalLLMProvider  # type: ignore
            return LocalLLMProvider(self.model)
        except Exception:
            # Graceful fallback when local provider is not available
            return MockLLMProvider()

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: Optional[int] = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Process prompt using the configured provider

        Args:
            prompt: Input prompt string
            max_tokens: Max response tokens
            temperature: Creativity control (0-1)
            timeout: Optional timeout override in seconds
            logprobs: Whether to return log-probabilities of output tokens
            top_logprobs: Number of top token alternatives to return (e.g. 5)

        Returns:
            Parsed JSON response or error dictionary
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        return self.provider.invoke(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=effective_timeout,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )


class OpenRouterProvider:
    """OpenRouter API provider (original implementation)"""

    def __init__(self, api_url: str, model: str, api_key: str, timeout: int):
        self.api_url = api_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.api_url:
            headers["X-Title"] = "Video Safety Agent"

        model_name = self.model
        if "googleapis.com" in self.api_url and model_name.startswith("google/"):
            model_name = model_name.replace("google/", "")

        effective_max_tokens = max(max_tokens, 2048)

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a content safety analyst. Return responses in JSON format.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
            "response_format": {"type": "json_object"},
        }

        # Google AI Studio does not support logprobs/top_logprobs in its OpenAI-compatible endpoint
        if logprobs and "googleapis.com" not in self.api_url:
            payload["logprobs"] = True
            if top_logprobs is not None:
                payload["top_logprobs"] = top_logprobs

        effective_timeout = timeout if timeout is not None else self.timeout

        response = None
        for attempt in range(2):
            try:
                response = requests.post(
                    self.api_url, headers=headers, json=payload, timeout=effective_timeout
                )

                if response.status_code != 200:
                    # If json_object format or logprobs failed, retry with clean basic payload
                    retry_payload = {
                        k: v for k, v in payload.items() 
                        if k not in ("response_format", "logprobs", "top_logprobs")
                    }
                    retry_resp = requests.post(
                        self.api_url, headers=headers, json=retry_payload, timeout=effective_timeout
                    )
                    if retry_resp.status_code == 200:
                        response = retry_resp
                    else:
                        if attempt == 0 and response.status_code in (429, 500, 502, 503, 504):
                            time.sleep(1.5)
                            continue
                        return {
                            "error": f"API error {response.status_code}: {response.text}",
                            "response": response.text,
                        }
                break  # Successful 200 response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as net_err:
                if attempt == 0:
                    time.sleep(1.5)
                    effective_timeout = max(effective_timeout, 45)
                    continue
                return {
                    "error": f"API connection error: {net_err}",
                    "response": str(net_err),
                }
            except Exception as e:
                return {
                    "error": f"Unexpected API error: {e}",
                    "response": str(e),
                }

        if response is None or response.status_code != 200:
            return {"error": "Failed to get a response from API", "response": ""}

        try:
            response_data = response.json()
            choice = response_data.get("choices", [{}])[0]
            msg = choice.get("message", {}) if isinstance(choice, dict) else {}
            content = msg.get("content")
            if not content and msg.get("reasoning"):
                content = msg.get("reasoning")
            if not isinstance(content, str):
                content = str(content or "")

            try:
                parsed_content = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                cleaned = content.strip()
                # Strip <think>...</think> tags if present from reasoning models
                if "<think>" in cleaned and "</think>" in cleaned:
                    end_think = cleaned.rfind("</think>")
                    cleaned = cleaned[end_think + len("</think>"):].strip()
                elif "<think>" in cleaned:
                    cleaned = cleaned.split("<think>", 1)[0].strip()

                if cleaned.startswith("```"):
                    nl = cleaned.find("\n")
                    if nl != -1:
                        cleaned = cleaned[nl + 1 :]
                    fence = cleaned.rfind("```")
                    if fence != -1:
                        cleaned = cleaned[:fence]
                    cleaned = cleaned.strip()
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    candidate = cleaned[start : end + 1]
                    try:
                        parsed_content = json.loads(candidate)
                    except Exception:
                        return {
                            "error": "Invalid JSON response from API",
                            "response": content,
                        }
                else:
                    return {
                        "error": "Invalid JSON response from API",
                        "response": content,
                    }

            if "usage" in response_data:
                parsed_content["_token_usage"] = response_data["usage"]

            if "logprobs" in choice and choice["logprobs"]:
                parsed_content["_logprobs"] = choice["logprobs"]

            parsed_content["_raw_response"] = response_data

            return parsed_content

        except json.JSONDecodeError:
            return {"error": "Invalid JSON response from API"}
        except KeyError:
            return {"error": "Unexpected API response format"}
        except Exception as e:
            return {"error": str(e)}


class MockLLMProvider:
    """Mock provider for development when no API keys are available"""

    def invoke(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = None,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "safety_assessment": "mock_analysis",
            "pred_is_harmful": False,
            "confidence": 0.5,
            "explanation": "This is a mock response for development purposes",
            "_mock": True,
        }
