"""
pipeline/llm/vllm_client.py — vLLM HTTP client via OpenAI-compatible endpoint.

Primary client for production use on the A100 node.
vLLM serves Qwen/Qwen3.5-9B via the OpenAI-compatible /v1/chat/completions endpoint.
This process communicates with it over HTTP — the model never loads into this process.

enable_thinking is always silently forced to False per pipeline design rules.
"""

from __future__ import annotations

import re
import time
import httpx

from pipeline.llm.base import LLMClient

# Qwen3.5-9B generates plain-text "Thinking Process:" preamble headers in
# non-thinking mode regardless of instructions.  Strip only the header line
# so the numbered analysis that follows it is preserved as useful content.
_THINKING_HEADER_RE = re.compile(
    r"^(?:thinking process|let me (?:think|analyze|look)|analysis|"
    r"step[- ]by[- ]step[^:\n]*)[\s:]*\n+",
    re.IGNORECASE,
)


def _strip_thinking_header(text: str) -> str:
    cleaned = _THINKING_HEADER_RE.sub("", text, count=1).strip()
    return cleaned if cleaned else text


class VLLMClient(LLMClient):
    """
    vLLM OpenAI-compatible HTTP client.

    Calls POST /v1/chat/completions on the vLLM server started by launch.sh.
    Supports both sync and async usage via httpx.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        model_id: str = "Qwen/Qwen3.5-9B",
        request_timeout: float = 300.0,
        **kwargs,
    ):
        self.base_url       = base_url.rstrip("/")
        self.model_id       = model_id
        self.request_timeout = request_timeout

    def generate(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,  # silently ignored — always False
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        payload = {
            "model":       self.model_id,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        _retries = 2
        _backoff  = 15.0
        for attempt in range(_retries + 1):
            try:
                with httpx.Client(timeout=self.request_timeout) as client:
                    resp = client.post(
                        f"{self.base_url}/v1/chat/completions",
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                break
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as exc:
                if attempt == _retries:
                    raise
                time.sleep(_backoff * (attempt + 1))

        raw = data["choices"][0]["message"]["content"].strip()
        return _strip_thinking_header(raw)

    async def generate_async(
        self,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        enable_thinking: bool = False,  # silently ignored — always False
    ) -> str:
        """Async variant for use in async pipeline stages."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        payload = {
            "model":       self.model_id,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        raw = data["choices"][0]["message"]["content"].strip()
        return _strip_thinking_header(raw)

    def is_available(self) -> bool:
        """Ping the vLLM health endpoint."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/health")
                return resp.status_code == 200
        except Exception:
            return False
