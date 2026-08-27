"""Google Gemini adapter using the google-genai SDK.

Gemini 3.7 Flash is the default configured model. Current Gemini 3.7/3.6 models
no longer accept legacy temperature/top_p/top_k sampling controls, so the adapter
advertises supports_temperature=False and uses thinking_level when configured.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ..schemas import PromptPart, ProviderCapabilities, ProviderManyResponse, ProviderResponse
from .base import ProviderAdapter


class GeminiProvider(ProviderAdapter):
    provider_id = "gemini"
    # google-genai already retries transient 5xx/429-class failures internally.
    # Keep exactly one retry owner to avoid multiplying API attempts.
    sdk_managed_retries = True
    capabilities = ProviderCapabilities(
        supports_vision=True,
        supports_multiple_images=True,
        supports_native_multiple_choices=True,
        supports_temperature=False,
        supports_thinking_level=True,
        max_native_choices=8,
    )

    def __init__(self, model: str, *, timeout_seconds: int = 120, request_retries: int = 3) -> None:
        super().__init__(model, timeout_seconds=timeout_seconds, request_retries=request_retries)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY tanımlı değil")
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("google-genai paketi kurulu değil") from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)

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
        return "".join(str(getattr(part, "text", "") or "") for part in parts).strip()

    def _config(self, *, candidate_count: int, thinking_level: str | None):
        from google.genai import types

        kwargs: dict[str, object] = {
            "candidate_count": candidate_count,
            "response_mime_type": "application/json",
            # AVMA-TSP never supplies callable tools. Explicitly disabling AFC
            # keeps direct generate_content calls stateless and removes the SDK
            # warning about AFC on Models.generate_content.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if thinking_level:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
        return types.GenerateContentConfig(**kwargs)

    def _generate_once(
        self,
        parts: Sequence[PromptPart],
        *,
        phase: str,
        temperature: float | None,
        thinking_level: str | None,
    ) -> ProviderResponse:
        response = self._client.models.generate_content(
            model=self.model,
            contents=self._contents(parts),
            config=self._config(candidate_count=1, thinking_level=thinking_level),
        )
        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None):
            text = self._candidate_text(response.candidates[0])
        return ProviderResponse(
            text=str(text or ""),
            provider=self.provider_id,
            model=self.model,
            phase=phase,
            usage=self._usage(response),
            raw_metadata={"candidate_count": 1, "thinking_level": thinking_level},
        )

    def _generate_many_native(
        self,
        parts: Sequence[PromptPart],
        *,
        count: int,
        phase: str,
        temperature: float | None,
        thinking_level: str | None,
    ) -> ProviderManyResponse:
        response = self._client.models.generate_content(
            model=self.model,
            contents=self._contents(parts),
            config=self._config(candidate_count=count, thinking_level=thinking_level),
        )
        usage = self._usage(response)
        candidates = getattr(response, "candidates", None) or []
        responses = [
            ProviderResponse(
                text=self._candidate_text(candidate),
                provider=self.provider_id,
                model=self.model,
                phase=f"{phase}_{index:02d}",
                usage=usage if index == 1 else {},
                raw_metadata={
                    "candidate_index": index,
                    "native_candidate_count": count,
                    "thinking_level": thinking_level,
                },
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
        return ProviderManyResponse(responses=responses, strategy="native_multiple_choices")
