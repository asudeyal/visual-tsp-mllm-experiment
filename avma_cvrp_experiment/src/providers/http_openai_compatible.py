"""Shared adapter for OpenAI-compatible multimodal chat-completions endpoints."""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Sequence

import requests

from ..schemas import PromptPart, ProviderCapabilities, ProviderResponse
from .base import ProviderAdapter


class OpenAICompatibleVisionProvider(ProviderAdapter):
    base_url: str
    api_key_env: str
    provider_id: str
    extra_headers: dict[str, str] = {}
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=False,
        supports_temperature=True,
        supports_thinking_level=False,
        max_native_choices=1,
    )

    def __init__(self, model: str, *, timeout_seconds: int = 120, request_retries: int = 3) -> None:
        super().__init__(model, timeout_seconds=timeout_seconds, request_retries=request_retries)
        self.api_key = os.getenv(self.api_key_env)
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} tanımlı değil")

    @staticmethod
    def _data_uri(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _content(self, parts: Sequence[PromptPart]) -> list[dict[str, object]]:
        content: list[dict[str, object]] = []
        for part in parts:
            if part.kind == "text":
                content.append({"type": "text", "text": str(part.value)})
            elif part.kind == "image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": self._data_uri(Path(part.value))},
                    }
                )
            else:
                raise ValueError(f"Bilinmeyen prompt part türü: {part.kind}")
        return content

    def _generate_once(
        self,
        parts: Sequence[PromptPart],
        *,
        phase: str,
        temperature: float | None,
        thinking_level: str | None,
    ) -> ProviderResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": "user", "content": self._content(parts)}],
        }
        if temperature is not None:
            payload["temperature"] = temperature

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"{self.provider_id} boş choices döndürdü")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        if isinstance(text, list):
            text = "".join(str(item.get("text", "")) for item in text if isinstance(item, dict))

        return ProviderResponse(
            text=str(text),
            provider=self.provider_id,
            model=self.model,
            phase=phase,
            usage=data.get("usage") or {},
            raw_metadata={
                "finish_reason": choices[0].get("finish_reason"),
                "request_id": response.headers.get("x-request-id"),
            },
        )
