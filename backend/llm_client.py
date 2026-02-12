from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()


class LLMClientError(RuntimeError):
    pass


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
        max_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens or int(os.getenv("ANTHROPIC_MAX_TOKENS", "700"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str | None = None, max_tokens: int | None = None) -> str:
        if not self.api_key:
            raise LLMClientError("ANTHROPIC_API_KEY no configurada.")

        effective_max_tokens = max_tokens or self.max_tokens
        payload = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            raise LLMClientError(
                f"Error HTTP llamando a Anthropic: {exc}. Body={detail[:800]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Error HTTP llamando a Anthropic: {exc}") from exc

        body = response.json()
        content = body.get("content", [])
        text_parts = [block.get("text", "") for block in content if block.get("type") == "text"]
        result = "\n".join(part for part in text_parts if part).strip()
        if not result:
            raise LLMClientError("Respuesta vacia desde Anthropic.")
        return result
