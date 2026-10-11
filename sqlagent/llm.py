"""A small chat client for free-tier LLM providers.

Models are named "provider:model", for example:

  groq:openai/gpt-oss-120b      Groq cloud (fast, free tier: 8K tokens/minute)
  gemini:gemini-3.5-flash       Google AI Studio free tier
  openrouter:<model>:free       OpenRouter free models
  ollama:qwen2.5-coder:7b       a local model - free, unlimited, data never leaves your laptop
  switchboard:smart             through the Switchboard gateway (my LLM gateway project)

Every provider except Ollama is called through the OpenAI-compatible
/chat/completions format. Ollama is called through its native /api/chat
endpoint because that is the only way to raise its context window (default
4K tokens - too small for a big database schema, and it truncates silently).

The client retries rate limits (429) by waiting as long as the provider asks,
and retries timeouts / 5xx errors with exponential backoff.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import httpx

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "switchboard": ("http://127.0.0.1:8000/v1", "SWITCHBOARD_API_KEY"),
    "ollama": ("http://127.0.0.1:11434", None),
}

MAX_WAIT_S = 600  # if a provider says "come back in more than 10 minutes", it's a daily limit: give up


class LLMError(RuntimeError):
    """The model could not be reached (bad key, daily limit hit, provider down, ...)."""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLM:
    model: str  # "provider:model"
    temperature: float = 0.0
    # Groq counts prompt + max_tokens against its 8K tokens-per-minute limit, so a big
    # reply budget can make a request "too large" before the model writes a word.
    max_tokens: int = 2048
    # Seconds to wait for a reply. None = 60 s for cloud APIs (they normally answer in a few
    # seconds, and a stuck request is better retried) and 300 s for a local model.
    timeout_s: float | None = None
    num_ctx: int = 8192  # Ollama only: context window in tokens
    extra: dict = field(default_factory=dict)  # extra request fields, e.g. {"reasoning_effort": "low"}
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        provider, sep, name = self.model.partition(":")
        if not sep or provider not in PROVIDERS:
            raise ValueError(f"model must look like 'provider:model' with provider in {sorted(PROVIDERS)}; got {self.model!r}")
        self.provider, self.model_name = provider, name
        base_url, key_env = PROVIDERS[provider]
        self.base_url = os.environ.get(f"{provider.upper()}_BASE_URL", base_url).rstrip("/")
        self.api_key = os.environ.get(key_env, "") if key_env else ""
        if key_env and not self.api_key and provider != "switchboard":
            raise LLMError(f"{key_env} is not set. Add it to .env (see .env.example).")
        if self.timeout_s is None:
            self.timeout_s = 300.0 if provider == "ollama" else 60.0
        self._http = httpx.Client(timeout=self.timeout_s)

    # ------------------------------------------------------------------ public

    def chat(self, messages: list[dict]) -> str:
        """Send the conversation, return the reply text."""
        if self.provider == "ollama":
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
            }
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                **self.extra,
            }
        data = self._post(url, payload)
        return self._parse(data)

    # ----------------------------------------------------------------- helpers

    def _post(self, url: str, payload: dict) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        transient_tries = 0
        while True:
            try:
                r = self._http.post(url, json=payload, headers=headers)
            except httpx.ConnectError as e:
                raise LLMError(f"cannot reach {self.provider} at {self.base_url} ({e}). Is it running?") from e
            except httpx.TimeoutException:
                r = None

            if r is not None and r.status_code == 200:
                return r.json()

            if r is not None and r.status_code == 429:
                wait = _retry_after(r)
                if wait > MAX_WAIT_S:
                    raise LLMError(f"{self.provider} rate limit: retry in {wait:.0f}s (daily limit?). {r.text[:200]}")
                time.sleep(wait)
                continue

            if r is not None and r.status_code == 413 and payload.get("max_tokens", 0) > 512:
                # Prompt + reply budget is over the per-minute token limit: ask for a shorter reply.
                payload["max_tokens"] //= 2
                continue

            transient = r is None or r.status_code >= 500
            if transient and transient_tries < 4:
                time.sleep(2 * 2**transient_tries)  # 2s, 4s, 8s, 16s
                transient_tries += 1
                continue

            detail = "timeout" if r is None else f"HTTP {r.status_code}: {r.text[:300]}"
            raise LLMError(f"{self.provider} call failed: {detail}")

    def _parse(self, data: dict) -> str:
        self.usage.calls += 1
        if self.provider == "ollama":
            self.usage.prompt_tokens += data.get("prompt_eval_count", 0) or 0
            self.usage.completion_tokens += data.get("eval_count", 0) or 0
            return (data.get("message") or {}).get("content", "") or ""
        u = data.get("usage") or {}
        self.usage.prompt_tokens += u.get("prompt_tokens", 0) or 0
        self.usage.completion_tokens += u.get("completion_tokens", 0) or 0
        choice = (data.get("choices") or [{}])[0]
        return (choice.get("message") or {}).get("content", "") or ""


def _retry_after(r: httpx.Response) -> float:
    """How long the provider wants us to wait. Groq sends `retry-after` in seconds."""
    try:
        return max(1.0, float(r.headers.get("retry-after", "")))
    except ValueError:
        return 20.0
