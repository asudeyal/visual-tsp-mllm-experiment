"""Google Gemini adapter using the google-genai SDK.

Gemini 3.8 Flash uses independent candidate calls in AVMA-CVRP. Gemini 3+
does not support candidate_count, temperature, top_p or top_k; this adapter
therefore exposes no native multiple-choice path and relies on thinking_level.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Sequence

from ..schemas import PromptPart, ProviderCapabilities, ProviderResponse
from .base import ProviderAdapter


class GeminiProvider(ProviderAdapter):
    provider_id = "gemini"
    sdk_managed_retries = True
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=False,
        supports_temperature=False,
        supports_thinking_level=True,
        max_native_choices=1,
    )

    def __init__(
        self,
        model: str,
        *,
        timeout_seconds: int = 120,
        request_retries: int = 3,
        media_resolution: str | None = None,
    ) -> None:
        super().__init__(
            model,
            timeout_seconds=timeout_seconds,
            request_retries=request_retries,
        )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY tanımlı değil")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai paketi kurulu değil") from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.media_resolution = media_resolution

    @staticmethod
    def _contents(parts: Sequence[PromptPart]) -> list[object]:
        from PIL import Image

        contents: list[object] = []
        for part in parts:
            if part.kind == "text":
                contents.append(str(part.value))
            elif part.kind == "image":
                contents.append(Image.open(Path(part.value)))
            else:
                raise ValueError(f"Bilinmeyen prompt part türü: {part.kind}")
        return contents

    @staticmethod
    def _usage(response: object) -> dict[str, object]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return {}
        result: dict[str, object] = {}
        for attr in (
            "prompt_token_count",
            "candidates_token_count",
            "total_token_count",
            "thoughts_token_count",
        ):
            value = getattr(usage, attr, None)
            if value is not None:
                result[attr] = value
        return result

    @staticmethod
    def _candidate_text(candidate: object) -> str:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        return "".join(
            str(getattr(part, "text", "") or "") for part in parts
        ).strip()

    @staticmethod
    def _media_resolution_enum(types, value: str | None):
        if value is None:
            return None
        mapping = {
            "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
            "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
            "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
        }
        try:
            return mapping[value]
        except KeyError as exc:
            raise ValueError(f"Desteklenmeyen Gemini media_resolution: {value}") from exc

    def _config(self, *, thinking_level: str | None):
        from google.genai import types

        kwargs: dict[str, object] = {
            "response_mime_type": "application/json",
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level
            )
        media_resolution = self._media_resolution_enum(
            types,
            self.media_resolution,
        )
        if media_resolution is not None:
            kwargs["media_resolution"] = media_resolution
        return types.GenerateContentConfig(**kwargs)

    def _generate_once(
        self,
        parts: Sequence[PromptPart],
        *,
        phase: str,
        temperature: float | None,
        thinking_level: str | None,
    ) -> ProviderResponse:
        from google.genai import errors

        max_retries = self.request_retries
        response = None
        for attempt in range(max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=self._contents(parts),
                    config=self._config(thinking_level=thinking_level),
                )
                break
            except errors.ServerError as error:
                transient = "503" in str(error) or "UNAVAILABLE" in str(error)
                if transient and attempt < max_retries - 1:
                    print(
                        "API yoğunluğu (503). 30 saniye bekleniyor... "
                        f"(Deneme {attempt + 1}/{max_retries})"
                    )
                    time.sleep(30)
                    continue
                raise

        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None):
            text = self._candidate_text(response.candidates[0])

        return ProviderResponse(
            text=str(text or ""),
            provider=self.provider_id,
            model=self.model,
            phase=phase,
            usage=self._usage(response),
            raw_metadata={
                "candidate_strategy": "independent_calls",
                "thinking_level": thinking_level,
                "media_resolution": self.media_resolution,
            },
        )
